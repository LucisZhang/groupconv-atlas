"""Offline microbenchmark receipt audit; no device execution or automatic roofline claim."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

MICRO_SOURCE_SHA256 = '8242bbe4412e26d94a407030fca6c5a90d47ad6cd3b282837031a3adc9b27b10'
FROZEN_TREE = '5b499cd77b1594e501fce8c8d457ec147ffe410ae5764fa6b1df1a4566934cbd'
FROZEN_FILE_COUNT = 71
OPERATIONS = ('logical_copy', 'eight_fma_chains')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value):
    return type(value) is int


def analyze_samples(records):
    require(len(records) == 1201, 'formal microbenchmark requires environment + 1200 samples')
    env = records[0]
    require(env.get('kind') == 'environment' and isinstance(env.get('gpu'), str) and env['gpu'], 'environment missing')
    require(integer(env.get('l2_bytes')) and env['l2_bytes'] > 0 and integer(env.get('sm_count')) and env['sm_count'] > 0, 'invalid GPU properties')
    require(env.get('clock_control') == 'NOT_CONTROLLED' and env.get('counter_traffic') == 'NOT_MEASURED', 'unexpected microbenchmark provenance')
    groups = {}
    for row in records[1:]:
        require(row.get('kind') == 'sample', 'foreign record')
        op, block, scale = row.get('operation'), row.get('block'), row.get('scale')
        require(op in OPERATIONS and integer(block) and block in (128, 256) and integer(scale) and scale in (1, 2), 'unknown configuration')
        b, s = row.get('batch'), row.get('sample')
        require(integer(b) and 0 <= b < 5 and integer(s) and 0 <= s < 30, 'sample index outside formal grid')
        group = groups.setdefault((op, block, scale), {})
        require((b, s) not in group, 'duplicate sample')
        require(row.get('configuration_check') == 'PASS' and row.get('validation_scope') == 'first_16_after_warmup', 'missing actual validation scope')
        n = max(64 << 20, env['l2_bytes'] * 4) * scale // 4 if op == 'logical_copy' else env['sm_count'] * block * 8
        expected = {'threads': n, 'buffer_bytes': n * 4,
                    'read_write_bytes': n * 8 if op == 'logical_copy' else 0,
                    'iterations': 0 if op == 'logical_copy' else 4096 * scale,
                    'fp32_fma_flops': 0 if op == 'logical_copy' else n * 4096 * scale * 16}
        require(all(integer(row.get(k)) and row[k] == v for k, v in expected.items()), 'work/byte count differs from reviewed source')
        value = row.get('t_device_us')
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'invalid device time')
        group[b, s] = row
    require(len(groups) == 8 and all(len(g) == 150 for g in groups.values()), 'configuration denominator incomplete')
    output = []
    for (op, block, scale), group in sorted(groups.items()):
        medians = [statistics.median(group[b, s]['t_device_us'] for s in range(30)) for b in range(5)]
        median = statistics.median(medians)
        row = group[0, 0]
        work = row['read_write_bytes'] if op == 'logical_copy' else row['fp32_fma_flops']
        output.append({'operation': op, 'block': block, 'scale': scale, 'samples': 150,
                       'batch_medians_us': medians, 'median_batch_medians_us': median,
                       'batch_cv': statistics.stdev(medians) / statistics.mean(medians),
                       'work_per_call': work, 'rate_per_second': work / (median * 1e-6),
                       'rate_kind': 'logical_read_plus_write_bytes_per_second' if op == 'logical_copy' else 'source_formula_fp32_fma_flops_per_second',
                       'buffer_bytes': row['buffer_bytes'], 'validation_scope': 'first_16_after_warmup'})
    return {'environment': env, 'configurations': output, 'configuration_count': 8,
            'sample_count': 1200, 'classification': 'DESCRIPTIVE_ONLY',
            'sustained_compute_peak_status': 'PENDING_BINARY_INSTRUCTION_REVIEW',
            'dram_bandwidth_status': 'NOT_MEASURED', 'roofline_status': 'NOT_IMPLEMENTED',
            'profile_analysis_status': 'NOT_IMPLEMENTED', 'triton_analysis_status': 'NOT_IMPLEMENTED'}


def timestamp(value):
    require(isinstance(value, str), 'timestamp required')
    result = datetime.datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'timezone required')
    return result


def validate_commands(m, parent):
    start, end = timestamp(m.get('started_utc')), timestamp(m.get('finished_utc'))
    require(start <= end, 'reversed run timestamps')
    entries = m.get('commands', [])
    required = {'microbench', 'ptx', 'sass', 'resources', 'nvcc-version',
                'cuobjdump-version', 'nvdisasm-version', 'ncu-version',
                'telemetry-before', 'telemetry-after', 'processes-before', 'processes-after'}
    require(len(entries) == len(required) and {e.get('name') for e in entries} == required, 'command set incomplete/duplicated')
    indexed = {e['name']: e for e in entries}
    for entry in entries:
        begin, finish = timestamp(entry.get('started_utc')), timestamp(entry.get('finished_utc'))
        require(start <= begin <= finish <= end, 'command timestamps outside run')
        require(isinstance(entry.get('command'), list) and entry['command'], 'command argv missing')
        status = entry.get('status'); rc = entry.get('returncode')
        require(status in ('PASS', 'FAIL', 'TOOL_UNAVAILABLE', 'TIMEOUT'), 'invalid command status')
        require((type(rc) is int and ((rc == 0) == (status == 'PASS'))) if status in ('PASS','FAIL') else rc is None, 'command exit status mismatch')
        for stream in ('stdout', 'stderr'):
            name = entry.get(stream)
            require(isinstance(name, str) and not Path(name).is_absolute(), 'log path required')
            path = (parent / name).resolve()
            require(path.is_relative_to(parent.resolve()) and path.is_file() and sha(path) == entry.get(stream + '_sha256'), 'command log hash/path mismatch')
    micro = indexed['microbench']
    require(micro['status'] == 'PASS' and micro['command'] == [m['binary_path']], 'microbench command/binary mismatch')
    require(micro['stdout_sha256'] == m['raw_sha256'], 'microbench stdout differs from raw')
    require(m.get('microbench_started_utc') == micro['started_utc'] and m.get('microbench_finished_utc') == micro['finished_utc'], 'microbench stage timestamps mismatch')
    for phase in ('before', 'after'):
        for kind in ('telemetry', 'processes'):
            entry = indexed[kind + '-' + phase]
            require(entry['command'][0] == 'nvidia-smi', 'observation command must be nvidia-smi')
            if kind == 'telemetry':
                require('--query-gpu=uuid,temperature.gpu,power.limit,power.draw,clocks.sm' in entry['command'], 'telemetry fields missing')
            else:
                require('--query-compute-apps=gpu_uuid,pid,process_name,used_memory' in entry['command'], 'process query missing')
            require(timestamp(entry['finished_utc']) <= timestamp(micro['started_utc']) if phase == 'before' else timestamp(entry['started_utc']) >= timestamp(micro['finished_utc']), 'observation on wrong side of microbench')
    return indexed


def audit(raw, manifest, snapshot, binary):
    raw, manifest, snapshot, binary = map(Path, (raw, manifest, snapshot, binary))
    m = json.loads(manifest.read_text())
    require(m.get('kind') == 'microbench_capture_v1' and type(m.get('returncode')) is int and m['returncode'] == 0 and m.get('smoke') is False, 'successful formal capture manifest required')
    require(m.get('raw_sha256') == sha(raw) and m.get('binary_sha256') == sha(binary), 'raw/binary binding mismatch')
    before, after = m.get('gpu_before', {}), m.get('gpu_after', {})
    require(before == after and isinstance(before.get('uuid'), str) and before['uuid'] not in ('', 'UNKNOWN') and before.get('driver_version'), 'same GPU UUID/driver before and after required')
    commands = validate_commands(m, manifest.parent)
    source = m['source']; files = source['source_files']
    require(len(files) == FROZEN_FILE_COUNT and source.get('source_tree_sha256') == FROZEN_TREE, 'requires complete frozen 031 source identity')
    require(m.get('source_after') == source and m.get('binary_sha256_after') == m['binary_sha256'], 'source/binary changed or missing post-run identity')
    require(files.get('csrc/cuda/microbench.cu') == MICRO_SOURCE_SHA256, 'unreviewed microbenchmark source formula')
    require(hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == source['source_tree_sha256'], 'source tree digest mismatch')
    for name, digest in files.items():
        path = (snapshot / name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(snapshot.resolve()) and path.is_file() and sha(path) == digest, 'snapshot file mismatch: ' + name)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    result = analyze_samples(records)
    require(result['environment']['gpu'] == before.get('name'), 'raw GPU name mismatch')
    artifacts = {}
    for name in ('ptx', 'sass', 'resources'):
        entry = m.get('artifacts', {}).get(name)
        command = commands[name]
        option = {'ptx':'--dump-ptx','sass':'--dump-sass','resources':'--dump-resource-usage'}[name]
        require(command['command'] == ['cuobjdump', option, m['binary_path']], 'assembly command targets another binary')
        if entry is None:
            require(command['status'] != 'PASS' or (manifest.parent / command['stdout']).stat().st_size == 0, 'successful assembly output omitted')
            artifacts[name] = {'status': 'NOT_PROVIDED'}
            continue
        require(command['status'] == 'PASS' and entry['path'] == command['stdout'] and entry['sha256'] == command['stdout_sha256'], 'assembly artifact/command mismatch')
        path = (manifest.parent / entry['path']).resolve()
        require(not Path(entry['path']).is_absolute() and path.is_relative_to(manifest.parent.resolve()) and path.is_file() and sha(path) == entry['sha256'], 'assembly artifact hash/path mismatch')
        artifacts[name] = {'status': 'HASH_VERIFIED_NOT_SEMANTICALLY_REVIEWED', 'sha256': sha(path)}
    result.update(audit_status='PASS', binding=m, artifacts=artifacts,
                  input_sha256={'raw': sha(raw), 'manifest': sha(manifest), 'binary': sha(binary)},
                  analyzer_sha256=sha(__file__), limitations=[
                      'Manifest consistency does not independently prove execution, exclusivity or binary build provenance.',
                      'Microbenchmark raw records lack UUID/source identities; capture wrapper binds them at run level.',
                      'Copy counts logical read plus write bytes; no DRAM counter traffic was measured.',
                      'Only first 16 outputs after warmup were checked, not every output or timed sample.',
                      'Five batches are descriptive; no WIN/LOSS or cross-run stability conclusion.',
                      'FMA work formula requires manual PTX/SASS loop and independent-chain review before use as a compute ceiling.'])
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'manifest', 'snapshot', 'binary', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(argv)
    require(not args.output.exists(), 'output must be new')
    result = audit(args.raw, args.manifest, args.snapshot, args.binary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
