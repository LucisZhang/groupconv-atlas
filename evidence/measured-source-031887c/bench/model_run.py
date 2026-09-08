"""Formal convolution comparison on 12 frozen actual model-layer shapes.

Eight no-bias geometries are measured; four biased source operations remain in
all reports as native UNSUPPORTED. This is not a model accuracy/latency benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from .api import ROOT, Shape, Native
from .run import atomic_json, source_identity, sha
from .cuda_run import (VARIANTS, shape_worker, validate_correctness, finalize,
                       kill_shape_group, terminate_run, expected_support, precision, identity_gpu, gpu_snapshot)
from scripts.export_model_shapes import freeze_record

PROTOCOL_HASH = 'b66ec7ff3f749c8149534bc74afff469e4798383b3a8930331918c2b5ddb6938'
REPORT_HASH = 'f0cfd98be7ad8ee61e90533d0f4410ddb691e42a348b157f98589c6b8b9efdf4'
RECEIPT_HASH = 'ec8899f8064ce6323b489329571fdd4515e36e27eaa1c4bfb70fadd30be8ce4b'
PREPARATION = ROOT / 'results/preparation-20260908'


def validate_report_rows(report, protocol):
    if report.get('kind') != 'model_shape_export' or report.get('schema_version') != 2:
        raise ValueError('unknown shape report kind/version')
    if report.get('protocol_sha256') != PROTOCOL_HASH or report.get('protocol_id') != protocol['protocol_id']:
        raise ValueError('shape report protocol mismatch')
    if report.get('torchvision_version') != '0.23.0' or report.get('device') != 'cpu' or report.get('weights') is not None:
        raise ValueError('shape report source environment mismatch')
    rows = report.get('shapes', [])
    if len(rows) != 12 or len({r.get('shape_id') for r in rows}) != 12:
        raise ValueError('require exactly 12 unique source operations')
    if len({r.get('source_operation_id') for r in rows}) != 12:
        raise ValueError('duplicate source operation identity')
    try:
        observed = json.loads(json.dumps([freeze_record(row) for row in rows]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('malformed source operation') from exc
    if observed != protocol['frozen_shapes']:
        raise ValueError('source bias, operation identity or geometry differs from freeze')
    if sum(row['bias'] is True for row in rows) != 4:
        raise ValueError('biased denominator must remain four')
    for row in rows:
        if row['source_operation_attributes']['bias'] != row['bias']:
            raise ValueError('source bias property mismatch')
        if row['bias']:
            if row['native_status'] != 'UNSUPPORTED' or row['native_reason'] != 'bias not implemented':
                raise ValueError('biased original operation cannot be measured')
        elif row['native_status'] != 'SUPPORTED' or row['shape_id'] != Shape.from_record(row).id:
            raise ValueError('no-bias shape identity/support mismatch')
    return rows


def validate_shape_report(path):
    protocol_path = ROOT / 'configs/model-shapes-protocol.json'
    if sha(protocol_path) != PROTOCOL_HASH:
        raise ValueError('model protocol hash differs from fixed b66e protocol')
    if sha(path) != REPORT_HASH:
        raise ValueError('shape report file differs from verified CPU export')
    if sha(PREPARATION / 'receipt.json') != RECEIPT_HASH:
        raise ValueError('preparation receipt differs from fixed source manifest')
    receipt = json.loads((PREPARATION / 'receipt.json').read_text())
    if receipt['report_sha256'] != REPORT_HASH or receipt['protocol_sha256'] != PROTOCOL_HASH:
        raise ValueError('preparation receipt binding mismatch')
    files = {name: sha(ROOT / name) for name in receipt['export_source_files']}
    source_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if files != receipt['export_source_files'] or source_hash != receipt['export_source_sha256']:
        raise ValueError('export source changed since actual CPU forward')
    report = json.loads(Path(path).read_text())
    rows = validate_report_rows(report, json.loads(protocol_path.read_text()))
    return report, rows, source_hash


def excluded_result(row):
    return {'shape': row, 'samples': [], 'rows': [
        {'shape_id': row['shape_id'], 'implementation': name,
         'status': 'UNSUPPORTED' if name.startswith('k') else 'NOT_RUN',
         'planned_exclusion': True,
         'reason': 'bias not implemented' if name.startswith('k') else 'baseline not measured: original layer is outside native bias contract'}
        for name in VARIANTS]}


def pending_result(row, status, reason):
    return {'shape': row, 'samples': [], 'rows': [
        {'shape_id': row['shape_id'], 'implementation': name, 'status': status, 'reason': reason}
        for name in VARIANTS]}


def validate_worker_result(result, record, protocol):
    """Reject partial samples or a PASS without both numerical reference gates."""
    if record['bias']:
        raise ValueError('biased source operation must never reach a shape worker')
    shape = Shape.from_record(record)
    rows = result.get('rows', [])
    if len(rows) != len(VARIANTS) or {r.get('implementation') for r in rows} != set(VARIANTS):
        raise ValueError('worker implementation coverage mismatch')
    for row in rows:
        name = row['implementation']
        supported = expected_support('cuda', shape, int(name[1:])) if name.startswith('k') else True
        if row.get('shape_id') != shape.id:
            raise ValueError('worker shape identity mismatch')
        if row.get('status') not in ('PASS', 'FAIL', 'UNSUPPORTED', 'OOM', 'NOT_RUN', 'TIMEOUT'):
            raise ValueError('unknown worker status')
        if row['status'] == 'UNSUPPORTED' and supported:
            raise ValueError('unexpected support rejection')
        if row['status'] != 'PASS':
            continue
        if not supported:
            raise ValueError('unsupported geometry returned PASS')
        full, points = row.get('error', {}), row.get('fp64_points', {})
        if full.get('passed') is not True or full.get('elements') != math.prod(shape.output):
            raise ValueError('missing full FP32 output evidence')
        if (points.get('passed') is not True or points.get('point_count') != 64
                or points.get('shape_id') != shape.id
                or points.get('reference') != 'independent_python_fp64_points'
                or len({tuple(p.get('position', ())) for p in points.get('points', [])}) != 64
                or len(points.get('points', [])) != points['point_count']
                or any(p.get('passed') is not True for p in points['points'])):
            raise ValueError('missing 64-point independent FP64 evidence')
        values = [v for v in result.get('samples', []) if v.get('implementation') == name]
        pairs = {(v['batch'], v['sample']) for v in values}
        expected = {(b, s) for b in range(protocol['batches']) for s in range(protocol['samples_per_batch'])}
        if len(values) != len(expected) or pairs != expected:
            raise ValueError('partial or duplicated formal samples')
        for value in values:
            if value.get('shape_id') != shape.id or any(not math.isfinite(value.get(k, float('nan'))) or value[k] <= 0
                    for k in ('t_device_op_graph_us', 't_api_us', 't_host_feed_us')):
                raise ValueError('invalid timing sample')
    result['shape'] = record  # Preserve original model/layer attributes in every shard.
    return result


def finalize_models(out, records, protocol, identity, started):
    proxies = [SimpleNamespace(id=r['shape_id']) for r in records]
    counts = finalize(out, proxies, protocol, 'FORMAL_MODEL_CONV_SHAPES', identity['source_tree_sha256'], started)
    rows = json.loads((out / 'bench.json').read_text())
    failure = any(r['status'] in ('FAIL', 'OOM', 'TIMEOUT') or
                  (r['status'] == 'NOT_RUN' and not r.get('planned_exclusion')) for r in rows)
    summary = json.loads((out / 'summary.json').read_text())
    summary.update(original_layer_denominator=12, native_eligible_layer_count=8,
                   native_bias_unsupported_layer_count=4, planned_unmeasured_baseline_rows=8,
                   bias_substitution=False, status='FAIL_OR_INCOMPLETE' if failure else 'PASS',
                   interpretation='convolution-only comparisons for eligible original no-bias operations; biased operations retained as UNSUPPORTED; no model accuracy or end-to-end speedup claim')
    atomic_json(out / 'summary.json', summary)
    return failure, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape-report', type=Path, required=True)
    parser.add_argument('--correctness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape-timeout', type=float, default=600)
    parser.add_argument('--budget-seconds', type=float, default=7200)
    parser.add_argument('--batches', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args(argv)
    if args.batches < 5 or args.samples < 30:
        parser.error('formal model measurements require at least 5 batches x 30 samples')
    if not all(math.isfinite(v) and v > 0 for v in (args.shape_timeout, args.budget_seconds)):
        parser.error('timeouts and budgets must be finite and positive')
    report, records, export_source_hash = validate_shape_report(args.shape_report)
    identity = source_identity()
    native = Native()
    library_hash = sha(native.path)
    validate_correctness(json.loads(args.correctness.read_text()), identity, library_hash)
    if not native.lib.gc_cuda_available():
        raise RuntimeError('native CUDA unavailable; no measurements executed')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('torch CUDA unavailable; no measurements executed')
    settings = precision(torch)
    protocol = json.loads((ROOT / 'configs/cuda-protocol.json').read_text())
    protocol.update(batches=args.batches, samples_per_batch=args.samples, fp64_points=64,
                    shape_timeout_seconds=args.shape_timeout, total_budget_seconds=args.budget_seconds)
    # Never reuse a directory, even an empty one, as a different run.
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    key = {'source': identity, 'library_sha256': library_hash, 'correctness_sha256': sha(args.correctness),
           'shape_report_sha256': sha(args.shape_report), 'shape_protocol_sha256': PROTOCOL_HASH,
           'export_source_sha256': export_source_hash, 'protocol': protocol,
           'gpu': identity_gpu(torch), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
           'cudnn': torch.backends.cudnn.version(), 'precision': settings, 'nvidia_smi': gpu_snapshot(),
           'measurement_kind': 'FORMAL_MODEL_CONV_SHAPES', 'source_operations': [r['source_operation_id'] for r in records]}
    atomic_json(out / 'run-key.json', key)
    atomic_json(out / 'model-shapes.json', report)
    started = time.monotonic()
    # Materialize the full denominator before creating any child process.
    for row in records:
        value = excluded_result(row) if row['bias'] else pending_result(row, 'NOT_RUN', 'not started')
        atomic_json(out / 'shapes' / (row['shape_id'] + '.json'), value)
    context = mp.get_context('spawn')
    active = None
    connections = ()
    active_row = None
    previous = signal.signal(signal.SIGTERM, terminate_run)
    try:
        for row in records:
            if row['bias']:
                continue
            active_row = row
            remaining = args.budget_seconds - (time.monotonic() - started)
            result = None
            if remaining > 0:
                parent, child = context.Pipe(duplex=False)
                connections = (parent, child)
                active = context.Process(target=shape_worker, args=(child, row, protocol, native.path))
                active.start(); child.close()
                if parent.poll(min(remaining, args.shape_timeout)):
                    try:
                        result = parent.recv()
                    except EOFError:
                        pass
                if result is None:
                    kill_shape_group(active)
                active.join(5)
                if active.is_alive():
                    kill_shape_group(active)
                parent.close(); connections = (); active = None
            if result is None:
                result = pending_result(row, 'TIMEOUT' if remaining > 0 else 'NOT_RUN',
                                        'shape worker timeout/crash' if remaining > 0 else 'total budget exhausted')
            else:
                try:
                    result = validate_worker_result(result, row, protocol)
                except (ValueError, KeyError, TypeError) as exc:
                    result = {'shape': row, 'samples': [], 'rejected_worker_raw_repr': repr(result),
                              'rows': pending_result(row, 'FAIL', str(exc))['rows']}
            atomic_json(out / 'shapes' / (row['shape_id'] + '.json'), result)
            finalize_models(out, records, protocol, identity, started)
            active_row = None
    except BaseException as exc:
        if active_row is not None:
            atomic_json(out / 'shapes' / (active_row['shape_id'] + '.json'),
                        pending_result(active_row, 'NOT_RUN', f'interrupted: {type(exc).__name__}'))
        atomic_json(out / 'interruption.json', {'status': 'INTERRUPTED', 'type': type(exc).__name__, 'reason': str(exc)})
        raise
    finally:
        kill_shape_group(active)
        for connection in connections:
            connection.close()
        signal.signal(signal.SIGTERM, previous)
        failure, _ = finalize_models(out, records, protocol, identity, started)
    return int(failure)


if __name__ == '__main__':
    raise SystemExit(main())
