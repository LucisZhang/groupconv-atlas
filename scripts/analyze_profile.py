"""Offline frozen-031 Nsight capture audit. Never launches ncu or a GPU."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_supplemental import native_receipt
from scripts.ncu_csv import is_wide, decode_wide
from bench.api import Shape, FIELDS
from bench.fp64_points import select_points
from bench.cuda_run import validate_correctness
from scripts.profile_session import planned_jobs, choose_metrics, metric_names, parse_counter_csv, KERNEL_IDENTIFIERS, validate_help

ADAPTER_SHA256='ac15dc74b4c76c6949b099c7ac7a4f25f99e6116e264e8016d0d45ccc5490c89'

TREE='5b499cd77b1594e501fce8c8d457ec147ffe410ae5764fa6b1df1a4566934cbd'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return json.loads(json.dumps(value))


def relocate(value, remote, local):
    path=Path(value)
    require(path.is_absolute() and path.is_relative_to(remote),'artifact outside declared remote session')
    dest=(local/path.relative_to(remote)).resolve()
    require(dest.is_relative_to(local.resolve()) and dest.is_file(),'missing or escaping downloaded artifact')
    return dest


def audit_adapter(session, remote, files, captures):
    """Require explicit reviewed execution adaptation for every wide capture."""
    path=session/'parser-adapter.json'
    require(path.is_file(),'wide captures require explicit parser adapter receipt')
    receipt=read(path)
    require(receipt.get('kind')=='ncu_raw_parser_adapter_v1' and receipt.get('scope')=='parse_counter_csv_only' and receipt.get('raw_csv_rewritten') is False,'invalid parser adapter scope')
    require(type(receipt.get('returncode')) is int and receipt['returncode']==0,'adapter execution incomplete')
    require(receipt.get('frozen_profile_sha256')==files['scripts/profile_session.py'],'adapter frozen runner differs')
    require(receipt.get('wrapper_sha256')==ADAPTER_SHA256==sha(session/'parser-adapter.py'),'unreviewed parser wrapper')
    require(receipt.get('helper_sha256')==sha(Path(__file__).with_name('ncu_csv.py'))==sha(session/'parser-helper.py'),'parser helper mismatch')
    args=receipt.get('argv');require(isinstance(args,list) and all(isinstance(x,str) for x in args),'invalid adapter argv')
    require(len(args)%2==0 and len(args[::2])==len(set(args[::2])),'duplicate adapter arguments')
    options=dict(zip(args[::2],args[1::2]))
    require(set(options).issubset({'--phase','--output','--library','--correctness','--protocol','--ncu','--seconds','--hourly-price','--currency'}) and options.get('--phase')=='collect' and options.get('--output')==str(remote),'adapter invocation mismatch')
    calls=receipt.get('parse_calls');require(isinstance(calls,list) and len(calls)==len(captures),'adapter parse denominator mismatch')
    for call,capture in zip(calls,captures):
        rawpath=relocate(capture['csv'],remote,session);raw=rawpath.read_text()
        require(call.get('status')=='PASS' and call.get('input_sha256')==sha(rawpath)==capture['csv_sha256'],'adapter parse input mismatch')
        require(call.get('metrics')==capture['selected_metrics'] and call.get('implementation')==capture['implementation'] and type(call.get('implementation')) is int,'adapter parse selection mismatch')
        parsed,metadata=decode_wide(raw,capture['selected_metrics'],KERNEL_IDENTIFIERS[capture['implementation']]) if is_wide(raw) else (parse_counter_csv(raw,capture['selected_metrics'],capture['implementation']),{'format':'legacy_long'})
        require(call.get('result')==parsed==capture['counters'] and call.get('metadata')==metadata,'adapter parse output mismatch')
        # The frozen target receipts bind hashes; argv paths bind actual invocation.
        command=next(row['command'] for row in read(session/'receipt.json')['commands'] if row['name']==capture['name'])
        for option in ('--library','--correctness'):
            require(options.get(option)==command[command.index(option)+1],'adapter target input argv mismatch')
    return {'receipt_sha256':sha(path),'wrapper_sha256':ADAPTER_SHA256,'helper_sha256':receipt['helper_sha256'],'scope':receipt['scope'],'raw_csv_rewritten':False}


def counters(raw, selected, impl):
    wide=is_wide(raw)
    parsed=decode_wide(raw,selected,KERNEL_IDENTIFIERS[impl])[0] if wide else parse_counter_csv(raw,selected,impl)
    header=None; identities=set(); seen=[]
    for fields in csv.reader(io.StringIO(raw)):
        if 'Metric Name' in fields and 'Kernel Name' in fields:header=fields;continue
        if header and len(fields)==len(header):
            row=dict(zip(header,fields))
            if row.get('Metric Name'):
                identities.add((row.get('Process ID'),row.get('ID'),row.get('Kernel Name')))
                seen.append(row['Metric Name'])
    if not wide: require(len(identities)==1 and len(seen)==len(selected) and set(seen)==set(selected),'extra launch/metric in raw CSV')
    metrics=parsed['metrics']
    for name in ('dram__bytes_read.sum','dram__bytes_write.sum'):
        require(metrics[name]['unit'] in ('byte','bytes'),'DRAM metric must use base bytes')
    require(metrics['sm__cycles_elapsed.avg']['unit'] in ('cycle','cycles'),'SM elapsed metric must use cycles')
    duration=metrics.get('gpu__time_duration.sum')
    seconds=None
    if duration:
        units={'second':1.,'seconds':1.,'msecond':1e-3,'usecond':1e-6,'nsecond':1e-9,'s':1.,'ms':1e-3,'us':1e-6,'ns':1e-9}
        require(duration['unit'] in units,'unknown profile duration unit')
        seconds=duration['value']*units[duration['unit']]
        require(math.isfinite(seconds) and seconds>0,'profile duration must be positive')
    traffic=sum(metrics[k]['value'] for k in ('dram__bytes_read.sum','dram__bytes_write.sum'))
    return parsed,{'dram_bytes':traffic,'profile_seconds':seconds,'dram_bytes_per_second':traffic/seconds if seconds else None,
                   'timing_status':'PROFILE_TIME_PRESENT' if seconds else 'NOT_ASSESSED'}


def correctness(evidence,shape,seed=20260908):
    record=shape.record() if isinstance(shape,Shape) else shape
    require(all(type(record.get(k)) is int for k in FIELDS),'correctness shape parameter types')
    shape=Shape.from_record(record)
    require(canonical(record)==canonical(shape.record()) and record.get('bias') is False,'correctness canonical shape mismatch')
    require(all(type(v) is int for key in ('input_dims','weight_dims','output_dims') for v in record[key]),'correctness dimension types')
    full=evidence['torch_cuda_fp32_full']; fp=evidence['independent_fp64_points']
    require(full.get('passed') is True and type(full.get('failed_elements')) is int and full['failed_elements']==0 and type(full.get('elements')) is int and full['elements']==math.prod(shape.output) and full.get('atol')==full.get('rtol')==1e-4,'full FP32 correctness incomplete')
    require(all(type(full.get(k)) in (int,float) and math.isfinite(full[k]) and full[k]>=0 for k in ('max_abs','max_rel','rms')) and full.get('epsilon')==1e-12 and full['rms']<=full['max_abs']+1e-12,'FP32 error summary inconsistent')
    require(fp.get('passed') is True and type(fp.get('failed_points')) is int and fp['failed_points']==0 and type(fp.get('point_count')) is int and fp['point_count']==64 and fp.get('shape_id')==shape.id and fp.get('atol')==fp.get('rtol')==1e-4,'FP64 correctness incomplete')
    require(fp.get('reference')=='independent_python_fp64_points' and fp.get('selection')=='boundaries_groups_seeded_v1' and type(fp.get('seed')) is int and fp['seed']==seed,'FP64 reference/seed mismatch')
    require(type(fp.get('output_count')) is int and fp['output_count']==math.prod(shape.output) and type(fp.get('full_output')) is bool and fp['full_output']==(64==math.prod(shape.output)) and fp.get('output_dims')==list(shape.output) and all(type(v) is int for v in fp['output_dims']),'FP64 aggregate shape mismatch')
    points=fp.get('points',[])
    require([p.get('position') for p in points]==[list(v) for v in select_points(shape,64,seed)],'FP64 point selection differs from frozen seed')
    require(len(points)==64 and len({tuple(p['position']) for p in points})==64,'FP64 points incomplete/duplicate')
    for p in points:
        require(len(p['position'])==4 and all(type(v) is int and 0<=v<d for v,d in zip(p['position'],shape.output)),'FP64 point out of bounds')
        require(p.get('passed') is True and p.get('finite') is True and all(type(p[k]) in (int,float) and math.isfinite(p[k]) for k in ('actual','expected_fp64')) and abs(p['actual']-p['expected_fp64'])<=1e-4+1e-4*abs(p['expected_fp64']),'FP64 inequality failed')

        absolute=abs(p['actual']-p['expected_fp64'])
        for field,value in [('absolute_error',absolute),('relative_error',absolute/max(abs(p['expected_fp64']),1e-12)),('threshold',1e-4+1e-4*abs(p['expected_fp64']))]:
            require(type(p.get(field)) in (int,float) and math.isfinite(p[field]) and math.isclose(p[field],value,rel_tol=1e-12,abs_tol=1e-15),'FP64 error detail inconsistent')

def derived(shape, rates):
    flops=2*math.prod(shape.output)*(shape.cin//shape.groups)*shape.r*shape.s
    return {**rates,'nominal_flops':flops,'nominal_flops_per_second':flops/rates['profile_seconds'] if rates['profile_seconds'] else None,
            'nominal_flops_per_dram_byte':flops/rates['dram_bytes'] if rates['dram_bytes']>0 else None,
            'intensity_status':'COMPUTED_NOMINAL' if rates['dram_bytes']>0 else 'NOT_ASSESSED_ZERO_TRAFFIC'}


def capture_commands(command,export,capture,job,selected,ncu):
    require(command[0]==ncu,'mixed profiler executable')
    prefix=[command[0],'--config-file','off','--profile-from-start','off','--replay-mode','kernel','--cache-control','all','--clock-control','none','--metrics',','.join(selected),'--launch-count','1','--export',capture['report']]
    require(command[:len(prefix)]==prefix,'capture replay/cache/launch command mismatch')
    target_args=command[len(prefix):]
    require(len(target_args)==15 and target_args[1:3]==['-m','bench.profile_cuda'],'unexpected target command')
    options=dict(zip(target_args[3::2],target_args[4::2]))
    require(options=={'--shape-id':job['shape_id'],'--impl':str(job['implementation']),'--library':options.get('--library'),'--correctness':options.get('--correctness'),'--protocol':options.get('--protocol'),'--output':capture['target_receipt']},'target argv mismatch')
    paths={k:options[k] for k in ('--library','--correctness','--protocol')}
    require(all(Path(v).is_absolute() for v in paths.values()),'target input paths must be absolute')
    require(export['command']==[command[0],'--config-file','off','--import',capture['report'],'--csv','--page','raw','--print-units','base'] and export['stdout']==capture['csv'] and export['stdout_sha256']==capture['csv_sha256'],'report/export/CSV mismatch')
    return paths


def target_environment(t, smi):
    require(t.get('torch')=='2.8.0+cu128' and t.get('cuda')=='12.8' and type(t.get('cudnn')) is int and t['cudnn']==91002,'frozen target runtime mismatch')
    builds={'groupconv-atlas ABI=1; CPU=f32,f64; layout=NCHW; OpenMP='+state+'; c1_tile=32; c2_tile=32; FP-contract=off (build flag required)' for state in ('enabled','disabled')}
    require(t.get('native_build') in builds,'native build identity mismatch')
    gpu=t.get('gpu',{});cap=gpu.get('capability')
    require(isinstance(gpu.get('name'),str) and bool(gpu['name'].strip()) and isinstance(gpu.get('uuid'),str) and gpu['uuid'] not in ('','UNKNOWN') and type(gpu.get('memory')) is int and gpu['memory']>0,'GPU identity/properties missing')
    require(isinstance(cap,list) and len(cap)==2 and all(type(v) is int and v>=0 for v in cap) and cap[0]>0 and isinstance(gpu.get('visible_devices'),str),'GPU capability/visibility missing')
    def values(field):return [v.strip() for v in re.findall(r'^\s*'+re.escape(field)+r'\s*:\s*(.+)$',smi,re.M)]
    uuids=values('GPU UUID');drivers=values('Driver Version');names=values('Product Name')
    def canonical_uuid(value):
        require(isinstance(value,str) and re.fullmatch(r'(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',value) is not None,'malformed GPU UUID')
        return value.removeprefix('GPU-').lower()
    require(len(uuids)==1 and canonical_uuid(uuids[0])==canonical_uuid(gpu['uuid']) and names==[gpu['name']] and len(drivers)==1 and re.fullmatch(r'\d+(?:\.\d+)+',drivers[0]) is not None,'nvidia-smi UUID/name/driver missing or inconsistent')
    return {k:t[k] for k in ('gpu','torch','cuda','cudnn','precision','native_build')}|{'driver_version':drivers[0],'nvidia_smi_uuid':uuids[0]}


def audit(session,remote,snapshot,library,validation):
    session,remote,snapshot,library,validation=map(Path,(session,remote,snapshot,library,validation))
    require(remote.is_absolute(),'remote session root must be absolute')
    session=session.resolve();snapshot=snapshot.resolve();library=library.resolve();validation=validation.resolve()
    inputs={}
    def bound(value,digest):
        path=relocate(value,remote,session);require(sha(path)==digest,'downloaded artifact SHA mismatch')
        inputs[str(path.relative_to(session))]=digest;return path
    receipt=read(session/'receipt.json');inputs['receipt.json']=sha(session/'receipt.json')
    require(receipt['kind']=='hardware_profile_session','wrong session kind')
    source=receipt['source'];files=source['source_files']
    require(len(files)==71 and source['source_tree_sha256']==TREE and hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()==TREE,'requires complete frozen 031 source identity')
    for name,digest in files.items():
        path=(snapshot/name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(snapshot.resolve()) and path.is_file() and sha(path)==digest,'frozen snapshot mismatch')
    root=Path(__file__).resolve().parents[1];dependencies={}
    for name,module in list(sys.modules.items()):
        if name=='scripts.profile_session' or name=='scripts' or name=='bench' or name.startswith('bench.'):
            path=Path(module.__file__).resolve();require(path.is_relative_to(root),'foreign analysis dependency')
            relative=str(path.relative_to(root));require(files.get(relative)==sha(path),'analysis dependency differs from frozen source: '+relative)
            dependencies[relative]=sha(path)
    protocol=read(snapshot/'configs/profile-protocol.json')
    require(receipt['protocol']==protocol and receipt['protocol_sha256']==sha(snapshot/'configs/profile-protocol.json'),'protocol binding mismatch')
    require(receipt['library_sha256']==sha(library) and receipt['native_correctness_sha256']==sha(validation),'native binary/correctness SHA mismatch')
    report=read(validation)
    native_receipt({'source':source,'library_sha256':sha(library),'correctness_sha256':sha(validation)},validation)
    require(report['source']['source_files']==files,'correctness source files mismatch')
    jobs,unsupported=planned_jobs(protocol,read(snapshot/'configs/atlas.json'))
    require(receipt['planned_jobs']==jobs and receipt['unsupported_jobs']==unsupported,'planned support matrix mismatch')
    base={'analysis_dependencies':dependencies,'python_version':platform.python_version(),'python_executable_sha256':sha(sys.executable),'source_tree_sha256':TREE,'library_sha256':sha(library),'correctness_sha256':sha(validation),'analyzer_sha256':sha(__file__),'nonfrozen_analysis_helpers':{'scripts/analyze_supplemental.py':sha(Path(__file__).with_name('analyze_supplemental.py')),'scripts/ncu_csv.py':sha(Path(__file__).with_name('ncu_csv.py'))},'session_status':receipt['status'],'phase':receipt['phase'],'unsupported_jobs':unsupported,'planned_count':len(jobs),'roofline_status':'NOT_ASSESSED','roofline_reason':'requires sustainable microbenchmark results and manual selected-SASS review; this analyzer does not accept or certify a compute ceiling'}
    if receipt['status']!='PASS' or receipt['phase']!='collect' or receipt.get('counter_gate')!='PASS':
        return {**base,'audit_status':'NOT_ASSESSED','reason':'requires successful collect, not a probe or failed session','input_sha256':inputs,'captures':[],'comparisons':[]}
    captures=receipt['captures']
    expected=[('counter-probe',{'shape_id':protocol['probe_shape_id'],'implementation':protocol['probe_implementation']})]+[(f"{i:02d}-{j['shape_id']}-k{j['implementation']}",j) for i,j in enumerate(jobs)]
    require(len(captures)==31 and receipt['completed_supported_combinations']==30,'capture denominator incomplete')
    commands=receipt['commands'];required={'ncu-version','ncu-help','nvidia-smi','sections','metrics'}|{n+s for n,_ in expected for s in ('','-export')}
    require(len(commands)==67 and {c['name'] for c in commands}==required,'command matrix incomplete/duplicate')
    by={c['name']:c for c in commands}
    for c in commands:
        require(c['status']=='PASS' and type(c['returncode']) is int and c['returncode']==0,'command failed')
        require(datetime.datetime.fromisoformat(c['started_utc']).tzinfo is not None and type(c['elapsed_seconds']) in (int,float) and math.isfinite(c['elapsed_seconds']) and c['elapsed_seconds']>=0,'command timing missing')
        for stream in ('stdout','stderr'):
            path=bound(c[stream],c[stream+'_sha256'])
            require('ERR_NVGPUCTRPERM' not in path.read_text(errors='replace') and '==ERROR==' not in path.read_text(errors='replace'),'profiler error log')
    ncu=by['ncu-version']['command'][0]
    query_commands={'ncu-version':[ncu,'--version'],'ncu-help':[ncu,'--help'],'nvidia-smi':['nvidia-smi','-q'],'sections':[ncu,'--config-file','off','--list-sections'],'metrics':[ncu,'--config-file','off','--query-metrics','--query-metrics-mode','all']}
    require(all(by[k]['command']==v for k,v in query_commands.items()),'query/version command mismatch')
    validate_help(bound(by['ncu-help']['stdout'],by['ncu-help']['stdout_sha256']).read_text())
    available=metric_names(bound(by['metrics']['stdout'],by['metrics']['stdout_sha256']).read_text())
    shapes={Shape.from_record(r).id:Shape.from_record(r) for r in read(snapshot/'configs/atlas.json')['shapes']}
    rows=[];environment=None;target_paths=None
    for capture,(name,job) in zip(captures,expected):
        require(capture['name']==name and all(capture[k]==v for k,v in job.items()) and capture['status']=='PASS','capture identity/order mismatch')
        selected,missing=choose_metrics(protocol,available,job['shape_id'],name=='counter-probe')
        require(capture['selected_metrics']==selected and capture['optional_metrics_unavailable']==missing,'metric selection mismatch')
        reportpath=bound(capture['report'],capture['report_sha256']);require(reportpath.stat().st_size>0,'empty actual ncu report')
        csvpath=bound(capture['csv'],capture['csv_sha256']);targetpath=bound(capture['target_receipt'],capture['target_receipt_sha256'])
        paths=capture_commands(by[name]['command'],by[name+'-export'],capture,job,selected,ncu)
        if target_paths is None: target_paths=paths
        require(paths==target_paths,'target input paths changed within session')
        t=read(targetpath);shape=shapes[job['shape_id']]
        require(t['kind']=='single_native_profile_target' and t['status']=='PASS' and canonical(t['shape'])==canonical(shape.record()) and t['implementation']==job['implementation'],'target identity mismatch')
        require(t['source']['source_files']==files and t['source']['source_tree_sha256']==TREE and t['library_sha256']==sha(library) and t['native_correctness_sha256']==sha(validation) and t['protocol']==protocol and t['protocol_sha256']==receipt['protocol_sha256'],'target source/binary/protocol mismatch')
        precision=t['precision'];legacy={'api':'allow_tf32','matmul':False,'cudnn':False};modern={k:'ieee' for k in ('global','matmul','cudnn','conv','rnn')}|{'api':'fp32_precision'}
        require(precision==modern or (precision==legacy and precision['matmul'] is False and precision['cudnn'] is False),'target strict FP32 mismatch')
        require(type(t['target_logical_calls']) is int and t['target_logical_calls']==1 and t['warmup_calls']==10 and t['stream']=='non-default' and t['expected_kernel_identifier']==KERNEL_IDENTIFIERS[job['implementation']],'target launch scope mismatch')
        require(t['gpu']==capture['gpu']==receipt['gpu'] and t['gpu'].get('uuid') not in (None,'UNKNOWN',''),'GPU binding mismatch')
        env=target_environment(t,bound(by['nvidia-smi']['stdout'],by['nvidia-smi']['stdout_sha256']).read_text())
        if environment is None:environment=env
        require(env==environment,'mixed target environments')
        for key in ('correctness_before_profile','correctness_after_profile'):correctness(t[key],t['shape'],protocol['seed'])
        require(type(t['process_id']) is int and t['process_id']>0,'target PID invalid')
        parsed,rates=counters(csvpath.read_text(),selected,job['implementation'])
        require(parsed==capture['counters'] and parsed['process_id']==str(t['process_id']),'CSV target PID/counter mismatch')
        csv_metadata=decode_wide(csvpath.read_text(),selected,KERNEL_IDENTIFIERS[job['implementation']])[1] if is_wide(csvpath.read_text()) else {'format':'legacy_long'}
        rows.append({'name':name,'raw_csv_metadata':csv_metadata,**job,'is_probe':name=='counter-probe','kernel_name':parsed['kernel_name'],'process_id':parsed['process_id'],'launch_id':parsed['kernel_id'],'metrics':parsed['metrics'],'unavailable_metrics':missing,**derived(shape,rates)})
    if any(is_wide(relocate(c['csv'],remote,session).read_text()) for c in captures):
        base['parser_adapter']=audit_adapter(session,remote,files,captures)
        inputs['parser-adapter.json']=sha(session/'parser-adapter.json')
    contrasts=[]
    for sid in protocol['shape_ids']:
        group={r['implementation']:r for r in rows if r['shape_id']==sid and not r['is_probe']}
        for b,c in ((0,1),(0,2),(0,3),(1,2)):
            if b in group and c in group:
                contrasts.append({'shape_id':sid,'baseline':b,'candidate':c,'profile_time_ratio':group[b]['profile_seconds']/group[c]['profile_seconds'],'dram_byte_ratio':group[b]['dram_bytes']/group[c]['dram_bytes'] if group[c]['dram_bytes'] else None,'interpretation':'single profiled capture per implementation; descriptive, no confidence or ordinary-latency claim'})
    return {**base,'audit_status':'PASS','environment':environment,'input_sha256':inputs,'captures':rows,'comparisons':contrasts,'limitations':['Opaque ncu report hash and export argv are checked; this offline tool cannot re-export the binary report to independently verify CSV derivation.','Target FP64 receipt inequalities checked, expected values not recomputed.','Run/target GPU identity and PID linkage do not independently establish execution; downloaded manifest and execution receipt remain required.','Nominal padding FLOPs are not dynamic instruction FLOPs; L2 sectors remain sectors, no L2 roofline conversion.']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','remote-root','snapshot','library','correctness','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv);require(not a.output.exists(),'output must be new')
    result=audit(a.session,a.remote_root,a.snapshot,a.library,a.correctness)
    a.output.mkdir(parents=True);(a.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=result['captures']
    if rows:
        fields=['name','shape_id','implementation','is_probe','profile_seconds','dram_bytes','dram_bytes_per_second','nominal_flops','nominal_flops_per_second','nominal_flops_per_dram_byte']
        with (a.output/'counters.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps({'audit_status':result['audit_status'],'roofline_status':result['roofline_status'],'capture_count':len(rows)}))
    return 0
if __name__=='__main__':raise SystemExit(main())
