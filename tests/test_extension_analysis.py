"""Synthetic CPU fixtures only; no GPU measurement claims."""
import copy
import math
import random
import numpy as np
import pytest
from bench.api import Shape
from scripts.analyze_extensions import compare, audit_shape, VARIANTS
PRECISION={'api':'allow_tf32','matmul':False,'cudnn':False}


def fixture():
    shape=Shape(1,4,4,7,9,3,3,4,1,1,1,1,1,1)
    rows=[];samples=[]
    for name in VARIANTS:
        rows.append({'shape_id':shape.id,'implementation':name,'status':'PASS',
                     'error':{'passed':True,'failed_elements':0,'elements':252,'atol':1e-4,'rtol':1e-4,'epsilon':1e-12,'max_abs':0.,'max_rel':0.,'rms':0.},
                     'fp64_points':{'passed':True,'failed_points':0,'point_count':64,'shape_id':shape.id,'atol':1e-4,'rtol':1e-4,
                                    'points':[{'passed':True,'finite':True,'actual':1.,'expected_fp64':1.,'position':[0,i//63,(i%63)//9,i%9]} for i in range(64)]}})
        for b in range(10):
            for s in range(30):
                order=list(VARIANTS);random.Random(f'20260908:{shape.id}:{b}').shuffle(order)
                samples.append({'shape_id':shape.id,'implementation':name,'batch':b,'sample':s,
                                'implementation_order':order.index(name),'repeats':1,'t_device_op_graph_us':float(b+1),'t_api_us':20.,'t_host_feed_us':1.})
    return shape,{'shape':shape.record(),'rows':rows,'samples':samples,'precision':dict(PRECISION)}

@pytest.mark.parametrize('a,b,label',[(2,1,'WIN'),(1,2,'LOSS'),(1,1,'TIE')])
def test_constant_ratios(a,b,label):
    result=compare([a]*10,[b]*10)
    assert result['ratio']==a/b and result['ci95']==[a/b,a/b]
    assert result['classification']==label

def test_short_batches_are_descriptive():
    result=compare([2]*5,[1]*5)
    assert result['classification']=='UNCERTAIN' and result['ci95'] is None

def test_paired_bootstrap_preserves_pairing():
    result=compare(np.arange(1,11)*2,np.arange(1,11))
    assert result['ci95']==[2.,2.]

def test_wide_interval_uncertain():
    assert compare([.2]*5+[5]*5,[1]*10)['classification']=='UNCERTAIN'

def test_uses_median_of_batch_medians_not_pooled():
    shape,data=fixture()
    # Every batch contains 15 low and 15 high samples. Varying spreads makes
    # the median of medians differ from the pooled median despite equal sizes.
    pairs=[(1,100)]*5+[(40,42)]*5
    for s in data['samples']:
        if s['implementation']=='k0':s['t_device_op_graph_us']=pairs[s['batch']][s['sample']//15]
    meds,_=audit_shape(data,shape,10,precision=PRECISION)
    pooled=np.median([s['t_device_op_graph_us'] for s in data['samples'] if s['implementation']=='k0'])
    assert np.median(meds['k0']['t_device_op_graph_us'])==45.75
    assert compare(meds['k0']['t_device_op_graph_us'],[1]*10)['ratio']==45.75
    # Separate odd/even fixture establishes the aggregation contract explicitly.
    assert pooled==41 and pooled != 45.75
    assert meds['k0']['t_device_op_graph_us']==[50.5]*5+[41.]*5

@pytest.mark.parametrize('change,match',[
    ('duplicate','duplicate'),('missing','count'),('nan','timing'),('foreign','foreign'),
    ('bad_fp64','FP64'),('false_unsupported','supported'),('geometry','canonical')])
def test_rejects_tampering(change,match):
    shape,data=fixture()
    if change=='duplicate':data['samples'].append(copy.deepcopy(data['samples'][0]))
    if change=='missing':data['samples'].pop()
    if change=='nan':data['samples'][0]['t_api_us']=math.nan
    if change=='foreign':data['samples'][0]['shape_id']='foreign'
    if change=='bad_fp64':data['rows'][0]['fp64_points']['points'][0]['actual']=2
    if change=='false_unsupported':data['rows'][0]['status']='UNSUPPORTED'
    if change=='geometry':data['shape']['h']=8
    with pytest.raises(ValueError,match=match):audit_shape(data,shape,10,precision=PRECISION)

def test_failed_partial_samples_preserved():
    shape,data=fixture();data['rows'][0]['status']='OOM'
    data['samples']=[s for s in data['samples'] if s['implementation']!='k0' or s['batch']==0]
    medians,statuses=audit_shape(data,shape,10,precision=PRECISION)
    assert 'k0' not in medians and statuses['k0']=='OOM'

@pytest.mark.parametrize('a,b',[([],[]),([1],[1,2]),([float('inf')],[1]),([0],[1])])
def test_invalid_statistics(a,b):
    with pytest.raises(ValueError):compare(a,b)


def incomplete_run(tmp_path,monkeypatch):
    """Synthetic file schema; native correctness verifier mocked, never claimed tested here."""
    import json
    import hashlib
    from pathlib import Path
    import scripts.analyze_extensions as module
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    root=Path(__file__).resolve().parents[1];snapshot=tmp_path/'snapshot';atlas=tmp_path/'atlas'
    config=json.loads((root/'configs/atlas.json').read_text());protocol=json.loads((root/'configs/cuda-protocol.json').read_text())
    for name,data in [('atlas',config),('cuda-protocol',protocol)]:put(snapshot/f'configs/{name}.json',data)
    files={f'configs/{name}.json':module.sha(snapshot/f'configs/{name}.json') for name in ('atlas','cuda-protocol')}
    import sys
    for name,module in list(sys.modules.items()):
        if (name=='bench' or name.startswith('bench.') or name=='scripts') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();relative=str(path.relative_to(root))
            destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes());files[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    import scripts.analyze_extensions as module
    source={'source_files':files,'source_tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
    correct=tmp_path/'correctness.json';put(correct,{'source':source})
    key={'source':source,'library_sha256':'synthetic','correctness_sha256':module.sha(correct),'shapes':[Shape.from_record(s).id for s in config['shapes']],
         'protocol':dict(protocol,batches=10),'precision':dict(PRECISION),'gpu':{'uuid':'SYNTHETIC-NOT-A-GPU'},'measurement_kind':'FORMAL'}
    put(atlas/'run-key.json',key);put(atlas/'provenance.json',key)
    rows=[]
    for record in config['shapes']:
        shape=Shape.from_record(record)
        records=[{'shape_id':shape.id,'implementation':n,'status':'NOT_RUN'} for n in VARIANTS]
        put(atlas/'shapes'/f'{shape.id}.json',{'shape':shape.record(),'rows':records,'samples':[]})
        rows.extend([dict(r,sample_count=0) for r in records])
    put(atlas/'bench.json',rows);(atlas/'samples.jsonl').write_text('')
    put(atlas/'summary.json',{'source_tree_sha256':source['source_tree_sha256'],'protocol':key['protocol'],'measurement_kind':'FORMAL',
                            'row_count':480,'expected_row_count':480,'shape_count':80,'completed_shape_count':80,'sample_count':0,'status_counts':{'NOT_RUN':480}})
    monkeypatch.setattr(module,'validate_correctness',lambda *args:None)
    return module,atlas,correct,snapshot,put

def test_complete_denominator_does_not_mean_measurement_pass(tmp_path,monkeypatch):
    module,atlas,correct,snapshot,_=incomplete_run(tmp_path,monkeypatch)
    result=module.analyze(atlas,correct,snapshot)
    assert result['audit_status']=='PASS' and result['measurement_status']=='INCOMPLETE'
    assert result['status_counts']=={'NOT_RUN':480}
    assert all(r['comparable']==0 and r['total_shapes']==80 for r in result['aggregates'])

@pytest.mark.parametrize('change,match',[('gpu','provenance'),('protocol','protocol'),('source','snapshot'),('shape','file set'),('correctness','digest'),('summary','denominator')])
def test_run_binding_rejection(tmp_path,monkeypatch,change,match):
    module,atlas,correct,snapshot,put=incomplete_run(tmp_path,monkeypatch)
    if change=='gpu':
        data=module.read(atlas/'provenance.json');data['gpu']['uuid']='OTHER';put(atlas/'provenance.json',data)
    if change=='protocol':
        data=module.read(atlas/'run-key.json');data['protocol']['batches']=5;put(atlas/'run-key.json',data)
    if change=='source':(snapshot/'configs/atlas.json').write_text((snapshot/'configs/atlas.json').read_text()+' ')
    if change=='shape':next((atlas/'shapes').glob('*.json')).unlink()
    if change=='correctness':correct.write_text(correct.read_text()+' ')
    if change=='summary':
        data=module.read(atlas/'summary.json');data['row_count']=479;put(atlas/'summary.json',data)
    with pytest.raises(ValueError,match=match):module.analyze(atlas,correct,snapshot)

@pytest.mark.parametrize('field,value', [('dtype','float16'),('layout','NHWC'),('bias',True),('shape_id','wrong'),('output_dims',[1,4,7,8]),('input_dims',[1,4,7,8])])
def test_canonical_metadata_rejected(field,value):
    shape,data=fixture();data['shape'][field]=value
    with pytest.raises(ValueError,match='canonical'):audit_shape(data,shape,10,precision=PRECISION)

@pytest.mark.parametrize('change,match',[('tf32','precision'),('different_api','differs'),('order','order'),('repeat_zero','repeats'),('repeat_bool','repeats'),('repeat_cap','repeats'),('repeat_changed','repeats')])
def test_readback_and_sampling_contract(change,match):
    shape,data=fixture()
    if change=='tf32':data['precision']['cudnn']=True
    if change=='different_api':data['precision']={k:'ieee' for k in ('global','matmul','cudnn','conv','rnn')}|{'api':'fp32_precision'}
    if change=='order':data['samples'][0]['implementation_order']=(data['samples'][0]['implementation_order']+1)%6
    if change=='repeat_zero':data['samples'][0]['repeats']=0
    if change=='repeat_bool':data['samples'][0]['repeats']=True
    if change=='repeat_cap':data['samples'][0]['repeats']=21
    if change=='repeat_changed':data['samples'][1]['repeats']=2
    with pytest.raises(ValueError,match=match):audit_shape(data,shape,10,precision=PRECISION)

def test_run_precision_rejected(tmp_path,monkeypatch):
    module,atlas,correct,snapshot,put=incomplete_run(tmp_path,monkeypatch)
    key=module.read(atlas/'run-key.json');key['precision']['matmul']=True;put(atlas/'run-key.json',key)
    with pytest.raises(ValueError,match='precision'):module.analyze(atlas,correct,snapshot)

def test_analysis_dependency_drift_rejected(tmp_path,monkeypatch):
    module,atlas,correct,snapshot,put=incomplete_run(tmp_path,monkeypatch)
    key=module.read(atlas/'run-key.json');key['source']['source_files']['bench/check.py']='0'*64;put(atlas/'run-key.json',key)
    with pytest.raises(ValueError,match='analysis dependency'):module.analyze(atlas,correct,snapshot)

def test_worst_comparable_keeps_shape_and_classification():
    from scripts.analyze_extensions import aggregate_comparisons
    rows=[{'baseline':'k0','candidate':'k1','boundary':'t_api_us','shape_id':sid,'baseline_status':'PASS',
           'candidate_status':status,'ratio':ratio,'classification':category}
          for sid,status,ratio,category in [('a','PASS',2.,'WIN'),('b','PASS',.5,'LOSS'),('c','OOM',None,'NOT_COMPARABLE')]]
    result=next(r for r in aggregate_comparisons(rows) if r['baseline']=='k0' and r['candidate']=='k1' and r['boundary']=='t_api_us')
    assert result['total_shapes']==80 and result['comparable']==2
    assert (result['worst_ratio'],result['worst_shape_id'],result['worst_classification'])==(.5,'b','LOSS')
    assert result['candidate_status_counts']=={'PASS':2,'OOM':1}


@pytest.mark.parametrize('field,value',[('max_abs',float('nan')),('max_rel',float('inf')),('rms',-1.),('rms',1.),('max_abs',True),('failed_elements',False),('elements',252.),('epsilon',0.)])
def test_full_fp32_summary_rejection(field,value):
    shape,data=fixture()
    data['rows'][0]['error'][field]=value
    with pytest.raises(ValueError):audit_shape(data,shape,10,precision=PRECISION)
