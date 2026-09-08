"""Offline rejection tests; synthetic fixtures do not certify GPU execution."""
import copy
import json
from pathlib import Path
import random
import pytest
from scripts import analyze_triton as a
from bench.api import Shape


def protocol():
    return a.tr.load_protocol(Path(__file__).resolve().parents[1]/'configs/triton-protocol.json')


def fixture_samples():
    p=protocol(); shape=Shape(cin=4,cout=4,groups=4,h=5,w=9)
    rows=[]
    for b in range(5):
        order=list(a.tr.BLOCKS); random.Random(f"{p['seed']}:{shape.id}:{b}").shuffle(order)
        for s in range(30): rows.append(dict(shape_id=shape.id,implementation='triton',block=128,
            batch=b,sample=s,repeats=2,implementation_order=order.index(128),
            t_device_op_graph_us=b+1,t_api_us=2*(b+1),t_host_feed_us=.5*(b+1)))
    return shape,p,rows


def test_median_of_batch_medians():
    shape,p,rows=fixture_samples()
    result=a.samples(rows,shape,p,'triton',list(a.tr.BLOCKS),128)
    assert result['t_device_op_graph_us']['batch_medians'] == [1,2,3,4,5]
    assert result['t_device_op_graph_us']['median_batch_medians'] == 3


@pytest.mark.parametrize('corruption',['delete','duplicate','foreign','order','repeat','nan','bool'])
def test_reject_sample_corruption(corruption):
    shape,p,rows=fixture_samples()
    if corruption=='delete': rows.pop()
    if corruption=='duplicate': rows[-1]=copy.deepcopy(rows[0])
    if corruption=='foreign': rows[0]['shape_id']='foreign'
    if corruption=='order': rows[0]['implementation_order']=99
    if corruption=='repeat': rows[0]['repeats']=21
    if corruption=='nan': rows[0]['t_device_op_graph_us']=float('nan')
    if corruption=='bool': rows[0]['batch']=False
    with pytest.raises(ValueError): a.samples(rows,shape,p,'triton',list(a.tr.BLOCKS),128)


def test_mapping_keeps_original_receipt_bytes(tmp_path):
    payload=b'{"path":"/remote/results/check/artifact.ptx"}'
    (tmp_path/'receipt.json').write_bytes(payload)
    mapping=a.Artifacts(tmp_path,'/remote/results')
    data=mapping.entry({'path':'/remote/results/receipt.json','sha256':a.sha(tmp_path/'receipt.json')})
    assert data['path'].startswith('/remote/')
    assert (tmp_path/'receipt.json').read_bytes()==payload
    with pytest.raises(ValueError): mapping.path('/etc/passwd')
    with pytest.raises(ValueError): mapping.path('/remote/results/../escape')
    (tmp_path/'link').symlink_to('/etc/passwd')
    with pytest.raises(ValueError): mapping.path('/remote/results/link')
    with pytest.raises(ValueError): mapping.entry({'path':'/remote/results/receipt.json','sha256':'0'*64})


def test_missing_frozen_source_denominator(tmp_path):
    with pytest.raises(ValueError,match='denominator'): a.validate_source({'source_files':{'one.py':'0'*64}},tmp_path)


def environment():
    return dict(torch='2.8.0+cu128',cuda='12.8',cudnn=91002,triton='3.4.0',torch_git='build',
        gpu=dict(name='test',uuid='GPU-00000000-0000-4000-8000-000000000001',memory=100,capability=[8,9],visible_devices='0'),
        driver_uuid_versions='GPU-00000000-0000-4000-8000-000000000001, 580.0')


@pytest.mark.parametrize('field',['cuda','cudnn','torch','triton','driver_uuid_versions'])
def test_runtime_drift(field):
    env=environment(); a.runtime(env,protocol()); env[field]='changed'
    with pytest.raises(ValueError): a.runtime(env,protocol())


def test_complete_split():
    root=Path(__file__).resolve().parents[1]
    split=a.tr.split_shapes(a.read(root/'configs/atlas.json'),a.read(root/'configs/split.json'))
    for shapes in split.values():
        assert len(shapes)==40 and sum(a.tr.supported(s) for s in shapes)==15
        with pytest.raises(ValueError): a.denominator({'denominator':{'all_shapes':15,'supported':15,'unsupported':0}},shapes)


def test_missing_candidate_cannot_freeze():
    root=Path(__file__).resolve().parents[1]
    tune=a.tr.split_shapes(a.read(root/'configs/atlas.json'),a.read(root/'configs/split.json'))['tuning']
    with pytest.raises(ValueError,match='candidate evidence set'): a.tr.select_blocks([],tune,protocol())


def test_missing_three_seed_receipt():
    shape=Shape(cin=4,cout=4,groups=4,h=5,w=9)
    with pytest.raises(ValueError,match='correctness/compile'):
        a.tr.validate_candidate({'block':128,'correctness':[{'seed':0}],'compile':[]},shape,protocol(),[0,7,20260908],True)


def test_cli_missing_artifacts_is_not_assessed(tmp_path):
    out=tmp_path/'analysis.json'
    assert a.main(['--results',str(tmp_path),'--remote-root','/remote/results','--snapshot',str(tmp_path),
                   '--library',str(tmp_path/'missing.so'),'--correctness',str(tmp_path/'correctness.json'),
                   '--output',str(out)]) == 2
    result=json.loads(out.read_text())
    assert result['analysis_status']=='NOT_ASSESSED' and result['comparisons']==[]


def test_inventory_preserves_failure_and_missing_denominator(tmp_path):
    root=Path(__file__).resolve().parents[1]
    shape=a.tr.split_shapes(a.read(root/'configs/atlas.json'),a.read(root/'configs/split.json'))['holdout'][0]
    folder=tmp_path/'triton-evaluate';folder.mkdir()
    job=folder/'job.json';job.write_text(json.dumps({'rows':[{'block':128,'status':'OOM','reason':'oom'}]}))
    (folder/'receipt.json').write_text(json.dumps({'status':'FAIL','jobs':[{'shape_id':shape.id,'suite':'triton',
        'path':'/remote/results/triton-evaluate/job.json','sha256':a.sha(job)}]}))
    result=a.inventory(tmp_path,'/remote/results')['evaluate']
    assert result['expected_jobs']==55 and result['observed_jobs']==1
    assert result['observed_receipt_status']=='FAIL'
    assert any(u['observed_rows'] and u['observed_rows'][0]['status']=='OOM' for u in result['units'])
    assert sum(u['matching_jobs']==0 for u in result['units'])==54


def test_reject_cross_candidate_samples():
    shape,p,rows=fixture_samples()
    for row in rows:
        row['block']=256
        order=list(a.tr.BLOCKS)
        random.Random(f"{p['seed']}:{shape.id}:{row['batch']}").shuffle(order)
        row['implementation_order']=order.index(256)
    with pytest.raises(ValueError,match='enclosing candidate'):
        a.samples(rows,shape,p,'triton',list(a.tr.BLOCKS),128)


def test_relative_results_normalized_before_artifact_hash(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'download'
    for phase in ('check','tune','evaluate'):
        folder=root/f'triton-{phase}';folder.mkdir(parents=True)
        (folder/'receipt.json').write_text(json.dumps({'source':{}}))
    def stop_after_hashes(source,snapshot):
        raise ValueError('reached source validator after all receipt hashes')
    monkeypatch.setattr(a,'validate_source',stop_after_hashes)
    with pytest.raises(ValueError,match='reached source validator'):
        a.analyze(Path('download'),'/remote/results',tmp_path,tmp_path/'lib.so',tmp_path/'check.json')


def test_uuid_prefix_only_normalization():
    env=environment();env['gpu']['uuid']=env['gpu']['uuid'].removeprefix('GPU-')
    a.runtime(env,protocol())
    for bad in ('bad','00000000-0000-4000-8000-000000000002'):
        env['gpu']['uuid']=bad
        with pytest.raises(ValueError):a.runtime(env,protocol())

def test_extra_driver_and_nonnumeric_version_reject():
    for suffix in ('\nGPU-00000000-0000-4000-8000-000000000001, 580.0','garbage'):
        env=environment();env['driver_uuid_versions']+=suffix
        with pytest.raises(ValueError):a.runtime(env,protocol())
