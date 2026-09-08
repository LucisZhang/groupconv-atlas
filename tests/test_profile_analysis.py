"""Synthetic offline counter fixtures; never GPU evidence."""
import csv
import io
import math
import pytest
from bench.api import Shape
from scripts.analyze_profile import counters, derived, relocate, correctness

METRICS=['dram__bytes_read.sum','dram__bytes_write.sum','sm__cycles_elapsed.avg','gpu__time_duration.sum']
def raw(time_unit='nsecond',time_value='2000',extra=None):
    f=io.StringIO();w=csv.writer(f)
    w.writerow(['ID','Process ID','Kernel Name','Metric Name','Metric Unit','Metric Value'])
    for metric,unit,value in zip(METRICS,['byte','byte','cycle',time_unit],['1024','1024','100',time_value]):
        w.writerow(['0','77','void naive()',metric,unit,value])
    if extra:w.writerow(extra)
    return f.getvalue()

@pytest.mark.parametrize('unit,value',[('second','.000002'),('msecond','.002'),('usecond','2'),('nsecond','2000')])
def test_time_units_and_same_capture_traffic(unit,value):
    _,rates=counters(raw(unit,value),METRICS,0)
    assert rates['profile_seconds']==pytest.approx(2e-6)
    assert rates['dram_bytes_per_second']==pytest.approx(1.024e9)
    model=derived(Shape(),rates)
    assert model['nominal_flops_per_second']==pytest.approx(model['nominal_flops']/2e-6)

@pytest.mark.parametrize('change', ['unknown_unit','scaled_bytes','negative','nan','duplicate','other_launch','other_kernel','wrong_cycles','missing','error'])
def test_invalid_counter_evidence(change):
    text=raw()
    if change=='unknown_unit':text=raw('fortnight','2')
    if change=='scaled_bytes':text=text.replace(',byte,',',Kbyte,')
    if change=='negative':text=raw(time_value='-2')
    if change=='nan':text=raw(time_value='nan')
    if change=='duplicate':text=raw(extra=['0','77','void naive()',METRICS[0],'byte','1'])
    if change=='other_launch':text=raw(extra=['1','77','void naive()','unexpected','byte','1'])
    if change=='other_kernel':text=text.replace('naive','register_outputs')
    if change=='wrong_cycles':text=text.replace(',cycle,',',byte,')
    if change=='missing':text='\n'.join(text.splitlines()[:-1])
    if change=='error':text+='\n==ERROR== denied\n'
    with pytest.raises(ValueError):counters(text,METRICS,0)

def test_zero_traffic_never_infinite_intensity():
    _,rates=counters(raw().replace(',1024',',0'),METRICS,0)
    result=derived(Shape(),rates)
    assert result['dram_bytes_per_second']==0
    assert result['nominal_flops_per_dram_byte'] is None
    assert result['intensity_status']=='NOT_ASSESSED_ZERO_TRAFFIC'

def test_probe_has_no_profile_time():
    text='\n'.join(raw().splitlines()[:-1])
    _,rates=counters(text,METRICS[:-1],0)
    assert rates['profile_seconds'] is None and rates['dram_bytes_per_second'] is None

def test_remote_relocation_no_basename_fallback(tmp_path):
    (tmp_path/'csv').mkdir();(tmp_path/'csv/x').write_text('x')
    from pathlib import Path
    assert relocate('/remote/session/csv/x',Path('/remote/session'),tmp_path)==tmp_path/'csv/x'
    for value in ('/other/csv/x','csv/x','/remote/session/../x'):
        with pytest.raises(ValueError):relocate(value,Path('/remote/session'),tmp_path)

def evidence():
    shape=Shape(1,4,4,7,9,3,3,4)
    value={'torch_cuda_fp32_full':{'passed':True,'failed_elements':0,'elements':math.prod(shape.output),'epsilon':1e-12,'max_abs':0.,'max_rel':0.,'rms':0.,'atol':1e-4,'rtol':1e-4},
           'independent_fp64_points':{'passed':True,'failed_points':0,'point_count':64,'shape_id':shape.id,'output_dims':list(shape.output),'output_count':math.prod(shape.output),'full_output':False,'reference':'independent_python_fp64_points','selection':'boundaries_groups_seeded_v1','seed':20260908,'atol':1e-4,'rtol':1e-4,
                                     'points':[{'position':list(pos),'passed':True,'finite':True,'actual':1.,'expected_fp64':1.,'absolute_error':0.,'relative_error':0.,'threshold':.0002} for pos in __import__('bench.fp64_points',fromlist=['select_points']).select_points(shape)]}}
    return shape,value

def test_correctness_scope_valid():
    shape,value=evidence();correctness(value,shape)

@pytest.mark.parametrize('change',['partial','duplicate','nan','wrong_number','wrong_shape'])
def test_correctness_counterexamples(change):
    shape,value=evidence();fp=value['independent_fp64_points']
    if change=='partial':value['torch_cuda_fp32_full']['elements']-=1
    if change=='duplicate':fp['points'][1]=fp['points'][0]
    if change=='nan':fp['points'][0]['actual']=float('nan')
    if change=='wrong_number':fp['points'][0]['actual']=100.
    if change=='wrong_shape':fp['shape_id']='wrong'
    with pytest.raises(ValueError):correctness(value,shape)

@pytest.mark.parametrize('change',[None,'launch_count','report_import','raw_hash','target_impl','binary_relative','profiler','replay','target_output'])
def test_capture_export_target_command_linkage(change):
    from scripts.analyze_profile import capture_commands
    from scripts.profile_session import capture_command
    job={'shape_id':'frozen-id','implementation':1}
    capture={'report':'/remote/reports/a.ncu-rep','csv':'/remote/csv/a.csv','csv_sha256':'csvhash','target_receipt':'/remote/targets/a.json'}
    target=['python','-m','bench.profile_cuda','--shape-id','frozen-id','--impl','1','--library','/build/lib.so','--correctness','/run/correctness.json','--protocol','/source/config.json','--output',capture['target_receipt']]
    command=capture_command('ncu',{'replay_mode':'kernel','cache_control':'all','clock_control':'none'},METRICS,capture['report'],target)
    export={'command':['ncu','--config-file','off','--import',capture['report'],'--csv','--page','raw','--print-units','base'],'stdout':capture['csv'],'stdout_sha256':'csvhash'}
    if change=='launch_count':command[command.index('--launch-count')+1]='2'
    if change=='report_import':export['command'][4]='/other/report'
    if change=='raw_hash':export['stdout_sha256']='wrong'
    if change=='target_impl':command[command.index('--impl')+1]='3'
    if change=='binary_relative':command[command.index('--library')+1]='relative.so'
    if change=='profiler':command[0]='other-ncu'
    if change=='replay':command[command.index('--replay-mode')+1]='application'
    if change=='target_output':command[-1]='/other/output'
    if change:
        with pytest.raises(ValueError):capture_commands(command,export,capture,job,METRICS,'ncu')
    else:
        assert capture_commands(command,export,capture,job,METRICS,'ncu')['--library']=='/build/lib.so'

@pytest.mark.parametrize('field,value',[('epsilon',1e-8),('max_abs',float('nan')),('max_rel',-1.),('rms',.1),('failed_elements',False),('elements',252.)])
def test_full_fp32_error_summary_rejected(field,value):
    shape,data=evidence();data['torch_cuda_fp32_full'][field]=value
    with pytest.raises(ValueError):correctness(data,shape)

@pytest.mark.parametrize('field,value',[('seed',17),('output_count',True),('output_dims',[1,4,7,8]),('full_output',True),('point_count',64.),('failed_points',False),('selection','explicit'),('reference','other')])
def test_fp64_aggregate_contract_rejected(field,value):
    shape,data=evidence();data['independent_fp64_points'][field]=value
    with pytest.raises(ValueError):correctness(data,shape)

@pytest.mark.parametrize('field',['absolute_error','relative_error','threshold'])
def test_fp64_error_details_rejected(field):
    shape,data=evidence();data['independent_fp64_points']['points'][0][field]=1.
    with pytest.raises(ValueError):correctness(data,shape)

@pytest.mark.parametrize('field,value',[('dtype','float16'),('bias',True),('groups',2.),('output_dims',[1,4,7,8]),('shape_id','different')])
def test_full_shape_record_contract(field,value):
    shape,data=evidence();record=shape.record();record[field]=value
    with pytest.raises(ValueError):correctness(data,record)

def test_zero_duration_rejected_before_contrasts():
    with pytest.raises(ValueError,match='duration'):counters(raw(time_value='0'),METRICS,0)

def test_duration_underflow_rejected_before_contrasts():
    with pytest.raises(ValueError,match='duration'):counters(raw(time_unit='nsecond',time_value='1e-320'),METRICS,0)


def environment_fixture():
    return {'gpu':{'name':'NVIDIA RTX 4090','uuid':'GPU-00000000-0000-4000-8000-000000000001','memory':24*1024**3,'capability':[8,9],'visible_devices':'UNSET'},
            'torch':'2.8.0+cu128','cuda':'12.8','cudnn':91002,'precision':{'api':'allow_tf32','matmul':False,'cudnn':False},
            'native_build':'groupconv-atlas ABI=1; CPU=f32,f64; layout=NCHW; OpenMP=enabled; c1_tile=32; c2_tile=32; FP-contract=off (build flag required)'},'Driver Version : 580.76.05\n    Product Name : NVIDIA RTX 4090\n    GPU UUID : GPU-00000000-0000-4000-8000-000000000001\n'


def test_target_environment_matches_real_field_schema():
    from scripts.analyze_profile import target_environment
    t,smi=environment_fixture()
    assert target_environment(t,smi)['driver_version']=='580.76.05'

@pytest.mark.parametrize('change',['torch','cuda','cudnn','native_build','empty_name','bad_cap','zero_mem','other_uuid','driver_missing','wrong_name'])
def test_target_environment_rejections(change):
    from scripts.analyze_profile import target_environment
    t,smi=environment_fixture()
    if change in ('torch','cuda','cudnn','native_build'):t[change]=''
    if change=='empty_name':t['gpu']['name']=''
    if change=='bad_cap':t['gpu']['capability']=[True,9]
    if change=='zero_mem':t['gpu']['memory']=0
    if change=='other_uuid':smi=smi.replace('GPU-00000000-0000-4000-8000-000000000001','GPU-other')
    if change=='driver_missing':smi='\n'.join(smi.splitlines()[1:])
    if change=='wrong_name':smi=smi.replace('NVIDIA RTX 4090','NVIDIA A100')
    with pytest.raises(ValueError):target_environment(t,smi)


def test_single_pass_abi_contract_rejected():
    from scripts.analyze_supplemental import contract_receipt
    with pytest.raises(ValueError):contract_receipt([{'case':'dtype','status':'PASS','expected_status':1,'actual_status':1}])

@pytest.mark.parametrize('change',['missing_contract','duplicate_record','count_mismatch','nonfinite_pass'])
def test_supplemental_native_gate_rejects_incomplete_evidence(tmp_path,monkeypatch,change):
    # Isolate added offline checks: existing frozen boundary validator is mocked.
    import sys,json
    from pathlib import Path
    import bench.cuda_run as cuda
    from scripts.analyze_profile import native_receipt,sha
    monkeypatch.setattr(cuda,'validate_correctness',lambda *a,**k:None)
    root=Path(__file__).resolve().parents[1]
    files={str(Path(m.__file__).resolve().relative_to(root)):sha(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('bench.') and getattr(m,'__file__',None)}
    source={'source_tree_sha256':'synthetic','source_files':files}
    expected=dict(dtype=1,rank=1,strides=2,misaligned=1,device=1,buffer_bytes=1,alias=1,groups=1,geometry=1,implementation=1)
    checks=[{'case':k,'expected_status':v,'actual_status':v,'status':'PASS'} for k,v in expected.items()]
    checks += [{'case':k,'status':'PASS'} for k in ('reject_nonfinite_nan','reject_nonfinite_inf','reject_nonfinite_-inf','python_shape_overflow','python_impl_overflow','python_nonintegral')]
    _,e=evidence();row={'backend':'cuda','status':'PASS','metrics':e['torch_cuda_fp32_full']}
    report={'kind':'correctness','source':source,'library':{'sha256':'binary'},'availability':{'cuda':{'available':True}},'limit':None,'counts':{'PASS':1,'FAIL':0},'records':[row],'contract_checks':checks}
    if change=='missing_contract':report['contract_checks']=checks[:1]
    if change=='duplicate_record':report['records']=[row,row];report['counts']['PASS']=2
    if change=='count_mismatch':report['counts']['PASS']=2
    if change=='nonfinite_pass':row['metrics']['rms']=float('inf')
    path=tmp_path/'receipt.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError):native_receipt({'source':source,'library_sha256':'binary','correctness_sha256':sha(path)},path)


def test_uuid_prefix_representation_and_strict_identity():
    from scripts.analyze_profile import target_environment
    t,smi=environment_fixture()
    t['gpu']['uuid']=t['gpu']['uuid'].removeprefix('GPU-')
    result=target_environment(t,smi)
    assert result['gpu']['uuid']==t['gpu']['uuid']
    assert result['nvidia_smi_uuid']=='GPU-'+t['gpu']['uuid']
    for bad in ('not-a-uuid','00000000-0000-4000-8000-000000000002'):
        t['gpu']['uuid']=bad
        with pytest.raises(ValueError):target_environment(t,smi)
