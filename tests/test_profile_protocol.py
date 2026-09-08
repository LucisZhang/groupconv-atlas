import csv
import io
import json

import pytest

from bench.api import ROOT
from bench.profile_cuda import select_shape
from scripts import profile_session as profile


def protocol():
    return profile.load_protocol(ROOT/'configs/profile-protocol.json')


def raw_csv(metrics=None, kernel='naive(gc_shape,...)', value='1,024', pid='123', kid='0'):
    metrics = metrics or protocol()['probe_metrics']
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Process ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
    for metric in metrics:
        writer.writerow([kid, pid, kernel, metric, 'cycle' if 'cycles' in metric else 'byte', value])
    return output.getvalue()


def test_exact_frozen_shape_matrix():
    p = protocol()
    jobs, unsupported = profile.planned_jobs(p, json.loads((ROOT/'configs/atlas.json').read_text()))
    assert len(jobs) == 30 and len(unsupported) == 2
    assert all(job['implementation'] == 2 for job in unsupported)
    assert len({(j['shape_id'], j['implementation']) for j in jobs}) == 30
    assert {select_shape(j['shape_id']).cin // select_shape(j['shape_id']).groups for j in unsupported} == {8, 256}


def test_unknown_shape_rejected_before_gpu_import():
    with pytest.raises(ValueError, match='frozen Atlas'):
        select_shape('missing')


def test_numerical_counter_gate():
    metrics = protocol()['probe_metrics']
    assert profile.parse_counter_csv(raw_csv(), metrics, 0)['metrics'][metrics[0]]['value'] == 1024
    text = raw_csv([metrics[0], metrics[1]], value='0')
    text += raw_csv([metrics[2]], value='3')
    assert profile.parse_counter_csv(text, metrics, 0)['metrics'][metrics[0]]['value'] == 0


@pytest.mark.parametrize('value', ['N/A', 'nan', 'inf', '-1', ''])
def test_nonnumeric_or_invalid_is_not_counter_pass(value):
    with pytest.raises(ValueError):
        profile.parse_counter_csv(raw_csv(value=value), protocol()['probe_metrics'], 0)


def test_launchstats_exit_zero_is_insufficient():
    with pytest.raises(ValueError, match='no requested'):
        profile.parse_counter_csv(raw_csv(['launch__registers_per_thread']), protocol()['probe_metrics'], 0)
    with pytest.raises(ValueError, match='missing'):
        profile.parse_counter_csv(raw_csv(protocol()['probe_metrics'][:1]), protocol()['probe_metrics'], 0)


def test_no_wrong_kernel_duplicate_or_multiple_target():
    metrics = protocol()['probe_metrics']
    for text in (raw_csv(kernel='spatial(...)'), raw_csv() + raw_csv(), raw_csv() + raw_csv(kid='1'),
                 raw_csv() + '\n==ERROR== ERR_NVGPUCTRPERM\n'):
        with pytest.raises(ValueError):
            profile.parse_counter_csv(text, metrics, 0)


def test_zero_cycles_does_not_open_gate():
    with pytest.raises(ValueError, match='cycles'):
        profile.parse_counter_csv(raw_csv(value='0'), protocol()['probe_metrics'], 0)


def test_metric_query_uses_exact_names_and_records_optional_gaps():
    p = protocol()
    available = profile.metric_names('\n'.join(p['overview_required'] + ['lts__t_sectors.sum.per_second']))
    assert 'lts__t_sectors.sum' not in available
    selected, missing = profile.choose_metrics(p, available, p['shape_ids'][0])
    assert selected == p['overview_required']
    assert 'lts__t_sectors.sum' in missing
    with pytest.raises(ValueError, match='unavailable'):
        profile.choose_metrics(p, set(), p['shape_ids'][0], True)


def test_command_and_installed_help_gate():
    p = protocol()
    command = profile.capture_command('ncu', p, p['probe_metrics'], 'test.ncu-rep', ['python', 'target.py'])
    for flag, value in (('--config-file', 'off'), ('--clock-control', 'none'),
                        ('--cache-control', 'all'), ('--profile-from-start', 'off'), ('--launch-count', '1')):
        assert command[command.index(flag) + 1] == value
    assert '--set' not in command
    profile.validate_help('\n'.join(profile.REQUIRED_OPTIONS))
    with pytest.raises(ValueError, match='required options'):
        profile.validate_help('--metrics --csv')


def test_reparented_worker_remains_tracked_and_pid_reuse_does_not(monkeypatch):
    tracked = {20: (10, 'old-start'), 30: (10, 'start-30')}
    monkeypatch.setattr(profile, 'process_snapshot', lambda: {
        20: (1, 'old-start'), 21: (20, 'child-start'), 30: (1, 'reused-start')})
    profile.track_descendants(10, tracked)
    assert tracked[20] == (1, 'old-start')
    assert tracked[21] == (20, 'child-start')
    assert tracked[30][1] == 'start-30'
