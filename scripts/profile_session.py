"""Bounded local ncu session: query -> one real counter gate -> 30 fixed captures.

No resource provisioning or permission/clock changes. Run on an already
authorized Linux GPU host. An old successful or failed probe is never reused.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from bench.api import Shape
from bench.check import expected_support
from bench.profile_cuda import KERNEL_IDENTIFIERS
from bench.run import atomic_json, sha, source_identity

REQUIRED_OPTIONS = ('--profile-from-start', '--replay-mode', '--cache-control', '--clock-control',
    '--metrics', '--launch-count', '--export', '--import', '--csv', '--page', '--print-units',
    '--query-metrics', '--query-metrics-mode', '--list-sections', '--config-file')


def load_protocol(path):
    p = json.loads(Path(path).read_text())
    if (p['replay_mode'] != 'kernel' or p['cache_control'] != 'all'
            or p['clock_control'] != 'none' or p['profile_from_start'] != 'off'
            or p['warmup'] != 10 or p['implementations'] != [0, 1, 2, 3]
            or len(p['shape_ids']) != 8 or len(set(p['shape_ids'])) != 8):
        raise ValueError('profile protocol controls changed')
    for key in ('query_timeout_seconds', 'probe_timeout_seconds', 'capture_timeout_seconds',
                'total_budget_seconds'):
        if type(p[key]) is not int or p[key] <= 0: raise ValueError('invalid timeout bound')
    if set(p['probe_metrics']) != {'dram__bytes_read.sum', 'dram__bytes_write.sum', 'sm__cycles_elapsed.avg'}:
        raise ValueError('probe must request actual DRAM and SM counters')
    return p


def planned_jobs(p, atlas):
    by_id = {Shape.from_record(row).id: Shape.from_record(row) for row in atlas['shapes']}
    if len(by_id) != len(atlas['shapes']) or any(s not in by_id for s in p['shape_ids']):
        raise ValueError('frozen shape set incomplete')
    supported, unsupported = [], []
    for sid in p['shape_ids']:
        for impl in p['implementations']:
            job = {'shape_id': sid, 'implementation': impl}
            (supported if expected_support('cuda', by_id[sid], impl) else unsupported).append(job)
    if len(supported) != p['expected_supported_combinations']:
        raise ValueError('supported combination count changed')
    return supported, unsupported


def metric_names(query):
    # Require token matches, not substring matches (sum must not match sum.per_second).
    return set(re.findall(r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*__[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', query))


def choose_metrics(p, available, shape_id, probe=False):
    required = p['probe_metrics'] if probe else p['overview_required']
    missing = [name for name in required if name not in available]
    if missing: raise ValueError('required hardware metrics unavailable: ' + ', '.join(missing))
    optional = [] if probe else p['overview_optional'] + (p['detail_optional'] if shape_id in p['detail_shape_ids'] else [])
    selected = list(dict.fromkeys(required + [m for m in optional if m in available]))
    return selected, [m for m in optional if m not in available]


def validate_help(help_text):
    missing = [option for option in REQUIRED_OPTIONS if option not in help_text]
    if missing: raise ValueError('installed ncu help lacks required options: ' + ', '.join(missing))


def parse_counter_csv(value, metrics, impl):
    """Validate one logical kernel's raw long-format metrics; reject fake PASS."""
    if 'ERR_NVGPUCTRPERM' in value or '==ERROR==' in value:
        raise ValueError('profiler error in exported data')
    reader = csv.reader(io.StringIO(value))
    header, records = None, []
    for fields in reader:
        if 'Metric Name' in fields and 'Metric Value' in fields and 'Kernel Name' in fields:
            header = fields
            continue
        if header and len(fields) == len(header):
            record = dict(zip(header, fields))
            if record.get('Metric Name') in metrics: records.append(record)
    if not records: raise ValueError('no requested numeric hardware metrics in CSV')
    kernels = {(r.get('Process ID'), r.get('ID'), r['Kernel Name']) for r in records}
    if len(kernels) != 1 or any(not r.get('ID') or not r.get('Process ID') for r in records):
        raise ValueError('expected exactly one identifiable profiled kernel')
    kernel_name = records[0]['Kernel Name']
    if KERNEL_IDENTIFIERS[impl] not in kernel_name:
        raise ValueError('wrong native kernel identity: ' + kernel_name)
    result = {}
    for record in records:
        metric = record['Metric Name']
        if metric in result: raise ValueError('duplicate requested metric in CSV')
        try: number = float(record['Metric Value'].strip().replace(',', ''))
        except ValueError: raise ValueError('nonnumeric counter value: ' + metric) from None
        if not math.isfinite(number) or number < 0:
            raise ValueError('invalid hardware metric: ' + metric)
        result[metric] = {'value': number, 'unit': record.get('Metric Unit', '')}
    if set(result) != set(metrics): raise ValueError('missing requested hardware metrics')
    if result['sm__cycles_elapsed.avg']['value'] <= 0:
        raise ValueError('SM elapsed cycles must be positive')
    if 'gpu__time_duration.sum' in result and result['gpu__time_duration.sum']['value'] <= 0:
        raise ValueError('profiled duration must be positive')
    return {'kernel_name': kernel_name, 'process_id': records[0]['Process ID'],
            'kernel_id': records[0]['ID'], 'metrics': result}


def process_snapshot():
    result = {}
    if not Path('/proc').exists(): return result
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            fields = (path/'stat').read_text().rpartition(')')[2].split()
            result[int(path.name)] = (int(fields[1]), fields[19])
        except (OSError, IndexError, ValueError): pass
    return result


def track_descendants(pid, tracked):
    current = process_snapshot()
    # Parent PID can change after reparenting; /proc starttime is the PID-reuse guard.
    roots = {pid} | {p for p, ident in tracked.items()
                    if p in current and current[p][1] == ident[1]}
    while True:
        expanded = roots | {p for p, (parent, _) in current.items() if parent in roots}
        if expanded == roots: break
        roots = expanded
    tracked.update({p: current[p] for p in roots if p in current})


def stop_tree(process, tracked):
    track_descendants(process.pid, tracked)
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        track_descendants(process.pid, tracked)
        if process.poll() is not None: break
        time.sleep(0.05)
    current = process_snapshot()
    for pid, ident in tracked.items():
        if pid in current and current[pid][1] == ident[1]:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    if process.poll() is None: process.kill()
    process.wait(timeout=5)


def capture_command(ncu, protocol, metrics, report, target):
    return [ncu, '--config-file', 'off', '--profile-from-start', 'off',
            '--replay-mode', protocol['replay_mode'], '--cache-control', protocol['cache_control'],
            '--clock-control', protocol['clock_control'], '--metrics', ','.join(metrics),
            '--launch-count', '1', '--export', str(report), *target]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--correctness', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/profile-protocol.json')
    parser.add_argument('--phase', choices=('probe', 'collect'), required=True)
    parser.add_argument('--ncu', default='ncu')
    parser.add_argument('--seconds', type=int, help='can only reduce the protocol wall-time budget')
    parser.add_argument('--hourly-price', type=float, help='optional confirmed compute rate; excludes storage/network')
    parser.add_argument('--currency', default='UNKNOWN')
    args = parser.parse_args(argv)
    p = load_protocol(args.protocol)
    seconds = min(args.seconds if args.seconds is not None else p['total_budget_seconds'], p['total_budget_seconds'])
    if seconds <= 0 or (args.hourly_price is not None and (not math.isfinite(args.hourly_price) or args.hourly_price < 0)):
        parser.error('positive budget and finite nonnegative price required')
    jobs, unsupported = planned_jobs(p, json.loads((ROOT/'configs/atlas.json').read_text()))
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    args.library = args.library.resolve(); args.correctness = args.correctness.resolve()
    args.protocol = args.protocol.resolve()
    os.chdir(ROOT)
    os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', LC_ALL='C')
    started = time.monotonic()
    receipt = {'kind': 'hardware_profile_session', 'phase': args.phase, 'status': 'RUNNING',
        'counter_gate': 'NOT_RUN', 'protocol': p, 'protocol_sha256': sha(args.protocol),
        'source': source_identity(), 'library_sha256': sha(args.library),
        'native_correctness_sha256': sha(args.correctness), 'commands': [], 'captures': [],
        'planned_jobs': jobs, 'unsupported_jobs': unsupported,
        'hourly_price': args.hourly_price, 'currency': args.currency,
        'cost_boundary': 'elapsed session compute estimate only; no provisioning, storage, network or idle-time charge claim'}
    def save():
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['estimated_compute_cost'] = (receipt['elapsed_seconds'] * args.hourly_price / 3600
                                               if args.hourly_price is not None else None)
        atomic_json(out/'receipt.json', receipt)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('session interrupted by signal ' + str(signum))
    previous = signal.signal(signal.SIGTERM, interrupted)

    def run(name, command, timeout, stdout_path=None):
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0: raise TimeoutError('session wall-time budget exhausted')
        stdout_path = stdout_path or out/'logs'/f'{name}.stdout.log'
        stderr_path = out/'logs'/f'{name}.stderr.log'
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        row = {'name': name, 'command': command,
               'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'stdout': str(stdout_path), 'stderr': str(stderr_path)}
        begin = time.monotonic(); process = None; tracked = {}
        try:
            with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
                deadline = begin + min(timeout, remaining)
                while process.poll() is None:
                    track_descendants(process.pid, tracked)
                    wait = deadline - time.monotonic()
                    if wait <= 0: raise subprocess.TimeoutExpired(command, min(timeout, remaining))
                    try: process.wait(timeout=min(0.2, wait))
                    except subprocess.TimeoutExpired: continue
                row.update(returncode=process.returncode, status='PASS' if process.returncode == 0 else 'FAIL')
        except BaseException as exc:
            row.update(status='TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'INTERRUPTED'
                       if isinstance(exc, KeyboardInterrupt) else 'FAIL', error=str(exc), returncode=None)
            raise
        finally:
            if process is not None: stop_tree(process, tracked)
            row['elapsed_seconds'] = time.monotonic() - begin
            row['stdout_sha256'] = sha(stdout_path) if stdout_path.exists() else None
            row['stderr_sha256'] = sha(stderr_path) if stderr_path.exists() else None
            receipt['commands'].append(row); save()
        if row['status'] != 'PASS': raise RuntimeError(name + ' failed; counter work stopped')
        combined = stdout_path.read_text(errors='replace') + '\n' + stderr_path.read_text(errors='replace')
        if 'ERR_NVGPUCTRPERM' in combined or '==ERROR==' in combined:
            raise RuntimeError(name + ' reported profiler error despite exit status')
        return stdout_path.read_text(errors='replace')

    def capture(job, name, metrics, missing, timeout):
        target_receipt = out/'targets'/f'{name}.json'
        target = [sys.executable, '-m', 'bench.profile_cuda', '--shape-id', job['shape_id'],
                  '--impl', str(job['implementation']), '--library', str(args.library),
                  '--correctness', str(args.correctness), '--protocol', str(args.protocol),
                  '--output', str(target_receipt)]
        report = out/'reports'/f'{name}.ncu-rep'; report.parent.mkdir(parents=True, exist_ok=True)
        run(name, capture_command(args.ncu, p, metrics, report, target), timeout)
        if not report.is_file() or not report.stat().st_size:
            raise RuntimeError('missing ncu report: ' + name)
        target_data = json.loads(target_receipt.read_text())
        if (target_data['status'] != 'PASS' or target_data['target_logical_calls'] != 1
                or target_data['shape']['shape_id'] != job['shape_id']
                or target_data['implementation'] != job['implementation']
                or target_data['source']['source_tree_sha256'] != receipt['source']['source_tree_sha256']
                or target_data['library_sha256'] != receipt['library_sha256']
                or target_data['native_correctness_sha256'] != receipt['native_correctness_sha256']
                or target_data['protocol_sha256'] != receipt['protocol_sha256']):
            raise RuntimeError('target correctness/source/library binding failed')
        for boundary in ('correctness_before_profile', 'correctness_after_profile'):
            evidence = target_data[boundary]
            if (evidence['torch_cuda_fp32_full']['passed'] is not True
                    or evidence['independent_fp64_points']['passed'] is not True
                    or evidence['independent_fp64_points']['point_count'] != p['fp64_points']):
                raise RuntimeError('target full FP32/independent FP64 evidence incomplete')
        csv_path = out/'csv'/f'{name}.csv'
        raw = run(name + '-export', [args.ncu, '--config-file', 'off', '--import', str(report),
                  '--csv', '--page', 'raw', '--print-units', 'base'], p['query_timeout_seconds'], csv_path)
        counters = parse_counter_csv(raw, metrics, job['implementation'])
        if counters['process_id'] != str(target_data['process_id']):
            raise RuntimeError('profiled PID does not match validated target process')
        data = {'name': name, **job, 'status': 'PASS', 'selected_metrics': metrics,
                'optional_metrics_unavailable': missing, 'counters': counters,
                'report': str(report), 'report_sha256': sha(report),
                'csv': str(csv_path), 'csv_sha256': sha(csv_path),
                'target_receipt': str(target_receipt), 'target_receipt_sha256': sha(target_receipt),
                'gpu': target_data['gpu']}
        receipt['captures'].append(data); save()
        return data

    try:
        save()
        run('ncu-version', [args.ncu, '--version'], p['query_timeout_seconds'])
        help_text = run('ncu-help', [args.ncu, '--help'], p['query_timeout_seconds'])
        validate_help(help_text)
        receipt['installed_help_required_options'] = list(REQUIRED_OPTIONS); save()
        run('nvidia-smi', ['nvidia-smi', '-q'], p['query_timeout_seconds'])
        run('sections', [args.ncu, '--config-file', 'off', '--list-sections'], p['query_timeout_seconds'])
        query = run('metrics', [args.ncu, '--config-file', 'off', '--query-metrics',
                    '--query-metrics-mode', 'all'], p['query_timeout_seconds'])
        available = metric_names(query)
        probe_metrics, missing = choose_metrics(p, available, p['probe_shape_id'], True)
        probe = capture({'shape_id': p['probe_shape_id'], 'implementation': p['probe_implementation']},
                        'counter-probe', probe_metrics, missing, p['probe_timeout_seconds'])
        receipt['counter_gate'] = 'PASS'; receipt['gpu'] = probe['gpu']; save()
        if args.phase == 'collect':
            for index, job in enumerate(jobs):
                metrics, missing = choose_metrics(p, available, job['shape_id'])
                data = capture(job, f"{index:02d}-{job['shape_id']}-k{job['implementation']}",
                               metrics, missing, p['capture_timeout_seconds'])
                if data['gpu'] != receipt['gpu']: raise RuntimeError('GPU identity changed within session')
            if len(receipt['captures']) != 1 + p['expected_supported_combinations']:
                raise RuntimeError('incomplete fixed profiling matrix')
        receipt.update(status='PASS', completed_supported_combinations=len(receipt['captures']) - 1,
                       interpretation=p['interpretation'])
    except BaseException as exc:
        receipt.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL',
                       reason=f'{type(exc).__name__}: {exc}', counter_work_stopped=True)
        if receipt['counter_gate'] != 'PASS': receipt['counter_gate'] = 'NOT_PROVEN'
    finally:
        signal.signal(signal.SIGTERM, previous); save()
    print(json.dumps({'status': receipt['status'], 'counter_gate': receipt['counter_gate'],
                      'output': str(out)}), flush=True)
    return 0 if receipt['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
