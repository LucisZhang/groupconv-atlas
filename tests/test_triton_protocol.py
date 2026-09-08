import copy
import json
from dataclasses import replace

import pytest

from bench.api import ROOT, Shape, inputs, error_metrics
from bench.triton_run import (load_protocol, split_shapes, summarize,
                             select_blocks, require_binding, validate_check_receipt, validate_frozen)
from bench.fp64_points import prepare_reference, check_points
from bench.run import sha
from backends.triton import supported, make_call, BLOCKS


def protocol():
    return load_protocol(ROOT/'configs/triton-protocol.json')


@pytest.mark.parametrize('n,c,g,h,w,expected', [
    (1, 4, 4, 1, 19, True), (3, 6, 3, 5, 9, True),
    (3, 6, 6, 5, 9, True), (1, 4, 1, 11, 1, True),
    (1, 6, 2, 5, 9, False), (1, 6, 4, 5, 9, False),
    (1, 0, 1, 5, 9, False), (1, 4, 0, 5, 9, False)])
def test_tail_and_group_contract(n, c, g, h, w, expected):
    assert supported(Shape(n=n, cin=c, cout=c, groups=g, h=h, w=w)) is expected


@pytest.mark.parametrize('changes', [dict(cout=8), dict(sh=2), dict(dh=2),
    dict(r=5), dict(ph=0), dict(n=True), dict(h=2**30)])
def test_rejected_geometry(changes):
    assert not supported(replace(Shape(cin=4, cout=4, groups=2), **changes))


def test_unsupported_does_not_import_gpu_runtime():
    with pytest.raises(ValueError, match='UNSUPPORTED'):
        make_call(Shape(cin=6, cout=6, groups=2), None, None, None, 128)


def test_block_bounds_precede_gpu_import():
    with pytest.raises(ValueError, match='block'):
        make_call(Shape(cin=4, cout=4, groups=2), None, None, None, 64)


def test_split_is_complete_and_has_no_leakage():
    atlas = json.loads((ROOT/'configs/atlas.json').read_text())
    split = json.loads((ROOT/'configs/split.json').read_text())
    result = split_shapes(atlas, split)
    assert [len(result[k]) for k in ('tuning', 'holdout')] == [40, 40]
    assert [sum(map(supported, result[k])) for k in ('tuning', 'holdout')] == [15, 15]
    split['holdout'][0] = split['tuning'][0]
    with pytest.raises(ValueError, match='partition'):
        split_shapes(atlas, split)


def samples(value=2.0):
    return [{'batch': b, 'sample': s, 't_device_op_graph_us': value,
             't_api_us': value * 2, 't_host_feed_us': value}
            for b in range(5) for s in range(30)]


def test_raw_grid_not_just_count():
    rows = samples()
    assert summarize(rows, protocol())['t_device_op_graph_us']['median_batch_medians'] == 2
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match='grid'):
        summarize(rows, protocol())
    rows = samples(); rows[0]['t_api_us'] = float('nan')
    with pytest.raises(ValueError, match='nonpositive'):
        summarize(rows, protocol())


def candidate(shape,block,tmp_path):
    """Synthetic CPU receipt fixture, explicitly not device-execution evidence."""
    import numpy as np
    p=protocol(); x,w=inputs(shape,p['seed'])
    prepared=prepare_reference(shape,x,w,seed=p['seed'],count=p['fp64_points'])
    actual=np.zeros(shape.output,dtype=np.float32)
    for pos,value in zip(prepared['positions'],prepared['expected_fp64']): actual[tuple(pos)]=value
    path=tmp_path/f'{shape.id}-{block}.ptx'; path.write_text('// synthetic unit-test PTX fixture\n')
    return {'shape_id':shape.id,'block':block,'status':'PASS','samples':samples(1.0 if block==256 else 2.0),
        'correctness':[{'seed':p['seed'],'torch_cuda_fp32_full':error_metrics(actual,actual),
                        'fp64':check_points(prepared,actual)}],
        'compile':[{'seed':p['seed'],'block':block,'num_warps':4,'num_stages':1,
            'enable_fp_fusion':False,'explicit_fma':True,'metadata':'synthetic test fixture',
            'first_call_jit_and_launch_us':1.0,'ptx_path':str(path),'ptx_sha256':sha(path)}]}


def test_selection_rejects_holdout_missing_and_failed_candidates(tmp_path):
    shapes = [Shape(cin=4, cout=4, groups=g) for g in (4, 2, 1)]
    rows = [candidate(shape,b,tmp_path) for shape in shapes for b in BLOCKS]
    selected, _ = select_blocks(rows, shapes, protocol())
    assert selected == {'1': 256, '2': 256, '4': 256}
    for bad in (rows[:-1], rows + [dict(rows[0], shape_id='holdout')]):
        with pytest.raises(ValueError, match='evidence set'):
            select_blocks(bad, shapes, protocol())
    bad = copy.deepcopy(rows); bad[0]['status'] = 'OOM'
    with pytest.raises(ValueError, match='forbids freeze'):
        select_blocks(bad, shapes, protocol())


@pytest.mark.parametrize('removed',['correctness','compile'])
def test_timing_and_pass_do_not_replace_candidate_evidence(tmp_path,removed):
    shapes=[Shape(cin=4,cout=4,groups=g,h=2,w=3) for g in (4,2,1)]
    rows=[candidate(s,b,tmp_path) for s in shapes for b in BLOCKS]
    del rows[0][removed]
    with pytest.raises(ValueError,match='evidence missing'):
        select_blocks(rows,shapes,protocol())


def test_frozen_evidence_closure_and_inner_environment(tmp_path,monkeypatch):
    from bench import triton_run as runner
    shapes=[Shape(cin=4,cout=4,groups=g,h=2,w=3) for g in (4,2,1)]
    rows=[candidate(s,b,tmp_path) for s in shapes for b in BLOCKS]
    p=protocol(); binding={'source':'test'}; env={'driver_uuid_versions':'uuid,580.65','cudnn':91002,'cuda':'12.8'}
    # Boundary receipt validation has separate tests; this fixture targets the tuning closure.
    monkeypatch.setattr(runner,'validate_check_receipt',lambda *args:None)
    check=tmp_path/'check.json'; check.write_text('{}')
    jobs=[]
    for s in shapes:
        path=tmp_path/f'{s.id}.json'
        path.write_text(json.dumps({'shape':s.record(),'suite':'triton','binding':binding,'environment':env,
            'device_correctness':'PASS','performance':'PASS','rows':[r for r in rows if r['shape_id']==s.id]}))
        jobs.append({'path':str(path),'sha256':sha(path),'suite':'triton','shape_id':s.id})
    selected,scores=select_blocks(rows,shapes,p)
    check_entry={'path':str(check),'sha256':sha(check)}
    frozen={'status':'PASS','kind':'triton_frozen','binding':binding,'environment':env,
        'selected':selected,'scores_graph_us':scores,'tuning_ids':[s.id for s in shapes],
        'holdout_performance_accessed':False,'evidence':jobs+[check_entry]+[
            {'path':r['compile'][0]['ptx_path'],'sha256':r['compile'][0]['ptx_sha256']} for r in rows]}
    receipt={'status':'PASS','kind':'triton_tune','binding':binding,'environment':env,
             'jobs':jobs,'check_report':check_entry}
    path=tmp_path/'frozen.json'
    def write_and_validate(value):
        path.write_text(json.dumps(value)); receipt['frozen_sha256']=sha(path)
        (tmp_path/'receipt.json').write_text(json.dumps(receipt))
        validate_frozen(value,path,binding,env,p,shapes)
    write_and_validate(frozen)
    bad=copy.deepcopy(frozen); bad['evidence']=[]
    with pytest.raises(ValueError,match='evidence set'): write_and_validate(bad)
    bad=copy.deepcopy(frozen); bad['holdout_performance_accessed']=True
    with pytest.raises(ValueError,match='holdout'): write_and_validate(bad)
    jobpath=tmp_path/f'{shapes[0].id}.json'; data=json.loads(jobpath.read_text())
    data['environment']['cudnn']=91100; jobpath.write_text(json.dumps(data))
    receipt['jobs'][0]['sha256']=sha(jobpath)
    frozen['evidence'][0]['sha256']=sha(jobpath)
    with pytest.raises(ValueError,match='environment'): write_and_validate(frozen)


@pytest.mark.parametrize('field',['cudnn','driver_uuid_versions','cuda'])
def test_actual_runtime_component_changes_reject_binding(field):
    env={'cudnn':91002,'driver_uuid_versions':'uuid,580.65','cuda':'12.8'}
    receipt={'status':'PASS','binding':{},'environment':env}
    changed=dict(env); changed[field]='changed'
    with pytest.raises(ValueError,match='environment'): require_binding(receipt,{},changed)


def test_frozen_binding_refuses_source_or_device_changes():
    receipt = {'status': 'PASS', 'binding': {'source': 'a'}, 'environment': {'uuid': '1'}}
    require_binding(receipt, {'source': 'a'}, {'uuid': '1'})
    with pytest.raises(ValueError, match='binding'):
        require_binding(receipt, {'source': 'b'}, {'uuid': '1'})
    with pytest.raises(ValueError, match='environment'):
        require_binding(receipt, {'source': 'a'}, {'uuid': '2'})


def test_pass_label_cannot_replace_boundary_evidence():
    receipt = {'status': 'PASS', 'binding': {}, 'environment': {},
               'kind': 'triton_check', 'device_correctness': 'PASS', 'jobs': []}
    with pytest.raises(ValueError, match='coverage'):
        validate_check_receipt(receipt, {}, {}, protocol())


@pytest.mark.parametrize('started', [False, True])
def test_isolated_cleans_interrupted_start(monkeypatch, started):
    """SIGTERM's SystemExit must unwind resources even inside process.start()."""
    from types import SimpleNamespace
    from bench import triton_run as runner
    events = []
    class Connection:
        closed = False
        def close(self): self.closed = True
    parent, child = Connection(), Connection()
    class Process:
        pid = None
        alive = False
        def start(self):
            if started: self.pid = 123456; self.alive = True
            raise SystemExit(143)
        def is_alive(self): return self.alive
        def kill(self): events.append('kill'); self.alive = False
        def join(self, timeout):
            assert self.pid is not None
            events.append('join')
    process = Process()
    context = SimpleNamespace(Pipe=lambda **kwargs: (parent, child), Process=lambda **kwargs: process)
    monkeypatch.setattr(runner.mp, 'get_context', lambda method: context)
    def killpg(pid, sig):
        assert pid == 123456
        events.append('killpg'); process.alive = False
    monkeypatch.setattr(runner.os, 'killpg', killpg, raising=False)
    with pytest.raises(SystemExit) as error:
        runner.isolated(None, (), 1)
    assert error.value.code == 143
    assert parent.closed and child.closed
    assert events == (['killpg', 'join'] if started else [])
