"""Synthetic CPU schemas only. These fixtures are not GPU execution evidence."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.analyze_hardware_extensions import analyze_samples, audit, sha


def records():
    env = {'kind':'environment','gpu':'SYNTHETIC_GPU','l2_bytes':1024,'sm_count':2,
           'clock_control':'NOT_CONTROLLED','counter_traffic':'NOT_MEASURED'}
    rows = [env]
    for op in ('logical_copy','eight_fma_chains'):
        for block in (128,256):
            for scale in (1,2):
                n = (64<<20)*scale//4 if op=='logical_copy' else 2*block*8
                for b in range(5):
                    for s in range(30):
                        rows.append(dict(kind='sample',operation=op,block=block,scale=scale,batch=b,sample=s,
                            threads=n,buffer_bytes=n*4,read_write_bytes=n*8 if op=='logical_copy' else 0,
                            fp32_fma_flops=n*4096*scale*16 if op!='logical_copy' else 0,
                            iterations=4096*scale if op!='logical_copy' else 0,t_device_us=10.,
                            configuration_check='PASS',validation_scope='first_16_after_warmup'))
    return rows


def test_descriptive_formula_and_scope():
    result=analyze_samples(records())
    assert result['sample_count']==1200 and len(result['configurations'])==8
    assert result['dram_bandwidth_status']=='NOT_MEASURED'
    assert result['sustained_compute_peak_status']=='PENDING_BINARY_INSTRUCTION_REVIEW'
    for row in result['configurations']:
        assert row['rate_per_second']==pytest.approx(row['work_per_call']/1e-5)


@pytest.mark.parametrize('change',['missing','duplicate','nan','zero','byte_count','fma_count','scope','smoke','bool_index'])
def test_reject_invalid_samples(change):
    rows=records()
    if change=='missing':rows.pop()
    if change=='duplicate':rows[2]=copy.deepcopy(rows[1])
    if change=='nan':rows[1]['t_device_us']=float('nan')
    if change=='zero':rows[1]['t_device_us']=0
    if change=='byte_count':rows[1]['read_write_bytes']+=1
    if change=='fma_count':rows[-1]['fp32_fma_flops']+=1
    if change=='scope':rows[1]['validation_scope']='full_output'
    if change=='smoke':rows=rows[:17]
    if change=='bool_index':rows[1]['batch']=False
    with pytest.raises(ValueError):analyze_samples(rows)


def bound_fixture(tmp_path, monkeypatch):
    # Unit fixture only: substitute frozen identity constants, never production CLI.
    import scripts.analyze_hardware_extensions as module
    raw=tmp_path/'raw.jsonl';raw.write_text('\n'.join(map(json.dumps,records()))+'\n')
    binary=tmp_path/'synthetic.bin';binary.write_bytes(b'NOT A REAL GPU BINARY')
    snapshot=tmp_path/'snapshot';path=snapshot/'csrc/cuda/microbench.cu';path.parent.mkdir(parents=True)
    path.write_bytes((Path(__file__).resolve().parents[1]/'csrc/cuda/microbench.cu').read_bytes())
    files={'csrc/cuda/microbench.cu':sha(path)}
    gpu={'name':'SYNTHETIC_GPU','uuid':'SYNTHETIC_UUID','driver_version':'SYNTHETIC'}
    m={'kind':'microbench_capture_v1','returncode':0,'smoke':False,'raw_sha256':sha(raw),
       'binary_sha256':sha(binary),'gpu_before':gpu,'gpu_after':copy.deepcopy(gpu),
       'source':{'source_files':files,'source_tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}}
    monkeypatch.setattr(module, 'FROZEN_FILE_COUNT', 1)
    monkeypatch.setattr(module, 'FROZEN_TREE', m['source']['source_tree_sha256'])
    m['source_after']=copy.deepcopy(m['source']);m['binary_sha256_after']=m['binary_sha256']
    m.update(binary_path=str(binary),started_utc='2026-09-08T00:00:00+00:00',finished_utc='2026-09-08T00:00:10+00:00',commands=[])
    specs=[(name,[tool,'--version']) for name,tool in [('nvcc-version','nvcc'),('cuobjdump-version','cuobjdump'),('nvdisasm-version','nvdisasm'),('ncu-version','ncu')]]
    specs += [(name,['cuobjdump',option,str(binary)]) for name,option in [('ptx','--dump-ptx'),('sass','--dump-sass'),('resources','--dump-resource-usage')]]
    for phase in ('before','after'):
        specs += [('telemetry-'+phase,['nvidia-smi','--query-gpu=uuid,temperature.gpu,power.limit,power.draw,clocks.sm','--format=csv']),('processes-'+phase,['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'])]
    specs.append(('microbench',[str(binary)]))
    for name,argv in specs:
        sec=7 if name.endswith('after') else 4 if name=='microbench' else 1
        out=raw if name=='microbench' else tmp_path/(name+'.txt')
        if out != raw:out.write_text('')
        err=tmp_path/(name+'.stderr.txt');err.write_text('')
        m['commands'].append(dict(name=name,command=argv,status='PASS',returncode=0,
          started_utc=f'2026-09-08T00:00:0{sec}+00:00',finished_utc=f'2026-09-08T00:00:0{sec+1}+00:00',
          stdout=out.name,stderr=err.name,stdout_sha256=sha(out),stderr_sha256=sha(err)))
    m.update(microbench_started_utc=m['commands'][-1]['started_utc'],microbench_finished_utc=m['commands'][-1]['finished_utc'])
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(m))
    return raw,manifest,snapshot,binary,m


def test_synthetic_binding_is_not_execution_proof(tmp_path,monkeypatch):
    raw,manifest,snapshot,binary,_=bound_fixture(tmp_path,monkeypatch)
    result=audit(raw,manifest,snapshot,binary)
    assert result['audit_status']=='PASS'
    assert result['artifacts']['sass']['status']=='NOT_PROVIDED'
    assert 'does not independently prove execution' in result['limitations'][0]


@pytest.mark.parametrize('change',['gpu','raw','binary','source','artifact_escape','failed_exit'])
def test_reject_binding_changes(tmp_path,monkeypatch,change):
    raw,manifest,snapshot,binary,m=bound_fixture(tmp_path,monkeypatch)
    if change=='gpu':m['gpu_after']['uuid']='OTHER'
    if change=='raw':raw.write_text(raw.read_text()+'\n')
    if change=='binary':binary.write_bytes(b'changed')
    if change=='source':(snapshot/'csrc/cuda/microbench.cu').write_text('changed')
    if change=='artifact_escape':m['artifacts']={'sass':{'path':'../escape','sha256':'x'}}
    if change=='failed_exit':m['returncode']=1
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):audit(raw,manifest,snapshot,binary)


@pytest.mark.parametrize('change',['bool_exit','missing_commands','missing_time','wrong_binary','stderr_changed','wrong_stage','missing_telemetry','failed_artifact','one_file_production'])
def test_reject_execution_receipt_gaps(tmp_path,monkeypatch,change):
    import scripts.analyze_hardware_extensions as module
    raw,manifest,snapshot,binary,m=bound_fixture(tmp_path,monkeypatch)
    by={r['name']:r for r in m['commands']}
    if change=='bool_exit':m['returncode']=False
    if change=='missing_commands':m['commands'].pop()
    if change=='missing_time':del by['sass']['started_utc']
    if change=='wrong_binary':by['sass']['command'][-1]='different.bin'
    if change=='stderr_changed':(tmp_path/by['sass']['stderr']).write_text('changed')
    if change=='wrong_stage':m['microbench_started_utc']=m['started_utc']
    if change=='missing_telemetry':by['telemetry-before']['command']=['nvidia-smi','--format=csv']
    if change=='failed_artifact':
        by['sass'].update(status='FAIL',returncode=1)
        m['artifacts']={'sass':{'path':by['sass']['stdout'],'sha256':by['sass']['stdout_sha256']}}
    if change=='one_file_production':
        monkeypatch.setattr(module,'FROZEN_FILE_COUNT',71)
        monkeypatch.setattr(module,'FROZEN_TREE','5b499cd77b1594e501fce8c8d457ec147ffe410ae5764fa6b1df1a4566934cbd')
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):audit(raw,manifest,snapshot,binary)
