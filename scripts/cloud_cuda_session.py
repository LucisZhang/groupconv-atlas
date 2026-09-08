"""Run one bounded CUDA validation/measurement session on a rented Linux GPU.

This script never creates resources or changes access settings. The caller must
set a provider shutdown deadline before running it and copy results before stop.
Every command has a timeout and a durable exit-status receipt.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
        return int(fields[1]), fields[19]
    except (OSError, IndexError, ValueError):
        return None


def stop_tree(process):
    """Stop only descendants of this command, including workers with setsid()."""
    identities = {int(path.name): value for path in Path('/proc').iterdir()
                  if path.name.isdigit() and (value := process_identity(int(path.name)))}
    descendants = {process.pid}
    while True:
        expanded = descendants | {pid for pid, (parent, _) in identities.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for pid in descendants:
        current = process_identity(pid)
        if pid in identities and current and current[1] == identities[pid][1]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['probe', 'validate', 'measure'], required=True)
    parser.add_argument('--seconds', type=int, default=10800)
    parser.add_argument('--counter-probe', action='store_true', help='single bounded probe only when the environment has new counter-access evidence')
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error('--seconds must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OMP_DYNAMIC', 'FALSE')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    started = time.monotonic()
    receipts = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt('session received termination signal')
    signal.signal(signal.SIGTERM, interrupted)

    def run(name, command, timeout=600, required=True):
        remaining = args.seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError('session wall-time budget exhausted')
        log = out / 'logs' / (name + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        record = {'name': name, 'command': command, 'required': required,
                  'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        begin = time.monotonic()
        print('START', name, flush=True)
        try:
            with log.open('w') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    returncode = process.wait(timeout=min(timeout, remaining))
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    stop_tree(process)
                    raise
            record.update(returncode=returncode,
                          status='PASS' if returncode == 0 else 'FAIL')
        except subprocess.TimeoutExpired:
            record.update(returncode=None, status='TIMEOUT')
        except KeyboardInterrupt:
            record.update(returncode=None, status='INTERRUPTED')
        except OSError as exc:
            log.write_text(str(exc) + '\n')
            record.update(returncode=None, status='NOT_RUN')
        record.update(elapsed_seconds=time.monotonic()-begin,
                      log=str(log.relative_to(out)),
                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        receipts.append(record)
        write_json(out / (args.phase + '-receipt.json'), {
            'phase': args.phase, 'status': 'RUNNING', 'commands': receipts})
        print(record['status'], name, 'seconds=' + str(round(record['elapsed_seconds'], 2)), flush=True)
        if record['status'] == 'INTERRUPTED':
            raise KeyboardInterrupt('command interrupted')
        if required and record['status'] != 'PASS':
            raise RuntimeError(name + ' failed; inspect ' + str(log))
        return record

    status, error = 'PASS', None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('preinstalled PyTorch cannot access CUDA')
        if args.phase == 'probe':
            run('occupancy', ['nvidia-smi'])
            run('gpu_identity', ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit', '--format=csv'])
            run('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])
            run('uptime', ['uptime'])
            run('date', ['date', '-Is'])
            for key, path in [('cpu_max', '/sys/fs/cgroup/cpu.max'), ('cpuset', '/sys/fs/cgroup/cpuset.cpus.effective')]:
                if Path(path).exists(): (out / (key + '.txt')).write_text(Path(path).read_text())
            for name in ('nvcc', 'cmake', 'g++', 'compute-sanitizer', 'ncu'):
                run(name + '_version', [name, '--version'], required=name != 'ncu')
            run('pip_freeze', [sys.executable, '-m', 'pip', 'freeze'])
            write_json(out / 'torch-environment.json', {
                'python': sys.version, 'torch': str(torch.__version__),
                'torch_commit': torch.version.git_version, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'device': torch.cuda.get_device_name(),
                'capability': list(torch.cuda.get_device_capability()),
                'torch_config': torch.__config__.show(),
                'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'exclusive_device': 'NOT_ESTABLISHED', 'clock_control': 'NOT_CONTROLLED'})
        elif args.phase == 'validate':
            arch = ''.join(map(str, torch.cuda.get_device_capability()))
            run('configure_build', ['./run.sh', 'cuda', '-DCMAKE_CUDA_ARCHITECTURES=' + arch,
                '-DCMAKE_CUDA_FLAGS=--ptxas-options=-v'], timeout=600)
            run('ctest', ['ctest', '--test-dir', 'build', '--output-on-failure'], timeout=600)
            run('pytest', [sys.executable, '-m', 'pytest', '-q', 'tests'], timeout=300)
            for tool in ('memcheck', 'racecheck', 'synccheck'):
                for suffix, extra in [('numerical', []), ('stress', ['--stress'])]:
                    run(tool + '_' + suffix, ['compute-sanitizer', '--tool', tool,
                        '--error-exitcode', '99', './build/native_cuda', *extra], timeout=600)
            run('correctness', [sys.executable, '-m', 'bench.check', '--backend', 'cuda',
                '--shape-set', 'course', '--output', str(out / 'correctness.json')], timeout=1200)
            if args.counter_probe:
                run('ncu_counter_probe', ['ncu', '--replay-mode', 'kernel', '--cache-control', 'all',
                    '--clock-control', 'none', '--launch-count', '1', '--metrics',
                    'dram__bytes_read.sum,dram__bytes_write.sum,sm__cycles_elapsed.avg',
                    '--export', str(out/'counter-probe'), './build/native_cuda',
                    '--profile', '--only-impl', '0'], timeout=120, required=False)
            else:
                write_json(out/'counter-status.json',{'status':'NOT_RUN',
                    'reason':'counter access needs new environment evidence; previous AutoDL host denied access; no repeated full collection'})
            run('environment', [sys.executable, '-m', 'scripts.probe', '--output', str(out / 'environment.json')])
        else:
            receipt = json.loads((out / 'validate-receipt.json').read_text())
            if receipt.get('status') != 'PASS':
                raise RuntimeError('validation phase has not passed')
            for shape_set in ('course', 'course-batch4', 'atlas'):
                remaining = int(args.seconds - (time.monotonic() - started))
                run('measure_' + shape_set, [sys.executable, '-m', 'bench.cuda_run',
                    '--output', str(out / shape_set), '--shape-set', shape_set,
                    '--correctness', str(out / 'correctness.json'), '--budget-seconds',
                    str(max(1, remaining - 10))], timeout=max(1, remaining))
    except BaseException as exc:
        status, error = 'FAIL', str(exc)
        print('SESSION_FAIL', error, flush=True)
    finally:
        write_json(out / (args.phase + '-receipt.json'), {
            'phase': args.phase, 'status': status, 'error': error,
            'elapsed_seconds': time.monotonic()-started, 'commands': receipts,
            'provider_shutdown': 'caller must verify provider state; script completion is not shutdown'})
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
