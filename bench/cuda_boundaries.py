"""Separate CUDA layout, CPU residency and eviction experiments; never merge with v1."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import random
import signal
import time
from pathlib import Path
import numpy as np
from .api import ROOT, Shape, Native, NativeError, inputs, error_metrics
from .cuda_run import precision, identity_gpu, gpu_snapshot, validate_correctness, kill_shape_group, terminate_run
from .check import expected_support
from .run import atomic_json, source_identity, sha
from .stats import describe

VARIANTS = ('k0','k1','k2','k3','torch_tuned_nchw','torch_tuned_channels_last')


def sample(call, torch, stream, evict=None):
    """The eviction attempt precedes the event and wall-clock intervals."""
    cache = getattr(call, '_boundary_graph', None)
    if cache is None:
        start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
        graph = torch.cuda.CUDAGraph()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            start.record(stream); output = call(); end.record(stream)
        graph.replay(); stream.synchronize()
        captured = output.clone(); eager = call(); stream.synchronize()
        if not torch.allclose(captured, eager, atol=1e-4, rtol=1e-4):
            raise RuntimeError('graph/eager output mismatch')
        cache = graph, start, end, output
        call._boundary_graph = cache
    graph, start, end, output = cache
    if evict: evict(); stream.synchronize()
    graph.replay(); stream.synchronize()
    device = start.elapsed_time(end)*1000
    if evict: evict(); stream.synchronize()
    begin = time.perf_counter_ns(); call(); stream.synchronize()
    return {'t_device_op_graph_us':device, 't_api_us':(time.perf_counter_ns()-begin)/1000}


def worker(connection, record, protocol, library):
    import os
    if hasattr(os, 'setsid'): os.setsid()
    rows, samples = [], []
    shape = Shape.from_record(record)
    try:
        import torch
        import torch.nn.functional as F
        from .fp64_points import prepare_reference, check_points
        settings = precision(torch)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = protocol['cudnn_benchmark_limit']
        native = Native(library); stream = torch.cuda.Stream()
        x, w = inputs(shape, protocol['seed'])
        reference_points = prepare_reference(shape,x,w,seed=protocol['seed'],count=protocol['fp64_points'])
        with torch.inference_mode(), torch.cuda.stream(stream):
            tx, tw = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
            y = torch.empty(shape.output,device='cuda',dtype=torch.float32)
            hx, hw = torch.from_numpy(x).pin_memory(), torch.from_numpy(w).pin_memory()
            hy = torch.empty(shape.output,dtype=torch.float32,pin_memory=True)
            dx, dw = torch.empty_like(tx), torch.empty_like(tw)
            kwargs = dict(stride=(shape.sh,shape.sw),padding=(shape.ph,shape.pw),
                          dilation=(shape.dh,shape.dw),groups=shape.groups)
            stream.synchronize(); begin=time.perf_counter_ns()
            ref_tensor = F.conv2d(tx,tw,**kwargs); stream.synchronize()
            reference_search=(time.perf_counter_ns()-begin)/1000
            ref = ref_tensor.cpu().numpy()
            clx, clw = tx.contiguous(memory_format=torch.channels_last),tw.contiguous(memory_format=torch.channels_last)
            eviction = torch.empty(protocol['eviction_bytes']//4,device='cuda',dtype=torch.float32)
            def evict(): eviction.fill_(1.2345)
            conversion_calls = {
                'input_nchw_to_channels_last': lambda:tx.contiguous(memory_format=torch.channels_last),
                'weight_nchw_to_channels_last': lambda:tw.contiguous(memory_format=torch.channels_last),
                'output_channels_last_to_nchw': lambda:F.conv2d(clx,clw,**kwargs)}
            stream.synchronize(); begin=time.perf_counter_ns()
            clout = conversion_calls['output_channels_last_to_nchw'](); stream.synchronize()
            channels_last_search=(time.perf_counter_ns()-begin)/1000
            conversion_calls['output_channels_last_to_nchw'] = lambda:clout.contiguous()
            conversions = {}
            for name, call in conversion_calls.items():
                for _ in range(protocol['warmup']): call()
                conversions[name] = [sample(call,torch,stream) for _ in range(protocol['conversion_samples'])]
            for name in VARIANTS:
                layout = 'channels_last' if name.endswith('channels_last') else 'NCHW'
                if name.startswith('k'):
                    impl = int(name[1])
                    if not expected_support('cuda',shape,impl):
                        rows.append({'implementation':name,'status':'UNSUPPORTED','reason':'native support contract'}); continue
                    def op(a,b,impl=impl): return native.cuda(shape,a,b,y,impl,stream.cuda_stream)
                else:
                    def op(a,b): return F.conv2d(a,b,**kwargs)
                source_x,source_w = (clx,clw) if layout=='channels_last' else (tx,tw)
                def ready(): return op(source_x,source_w)
                def caller_nchw():
                    a,b = (tx.contiguous(memory_format=torch.channels_last),tw.contiguous(memory_format=torch.channels_last)) if layout=='channels_last' else (tx,tw)
                    return op(a,b).contiguous()
                def host_to_host():
                    dx.copy_(hx,non_blocking=True); dw.copy_(hw,non_blocking=True)
                    a,b = (dx.contiguous(memory_format=torch.channels_last),dw.contiguous(memory_format=torch.channels_last)) if layout=='channels_last' else (dx,dw)
                    hy.copy_(op(a,b).contiguous(),non_blocking=True)
                    return hy
                calls = {'target_layout_ready':ready,'caller_nchw':caller_nchw,'host_to_host':host_to_host}
                stream.synchronize(); begin=time.perf_counter_ns(); actual=ready(); stream.synchronize()
                first=(time.perf_counter_ns()-begin)/1000
                correctness = {}
                for residency,call in calls.items():
                    actual=call(); stream.synchronize(); arr=actual.cpu().numpy()
                    full=error_metrics(arr,ref); point=check_points(reference_points,arr)
                    correctness[residency]={'fp32_full':full,'fp64_points':point}
                    if not full['passed'] or not point['passed']: raise RuntimeError(f'{name}/{residency} incorrect')
                    for _ in range(protocol['warmup']): call()
                    if residency!='host_to_host': sample(call,torch,stream)
                row={'implementation':name,'status':'PASS','layout':layout,
                     'strides':{'input':list(source_x.stride()),'weight':list(source_w.stride())},
                     'correctness':correctness,'first_variant_call_us_after_framework_setup':first,
                     'output_allocation':'reused' if name.startswith('k') else 'allocated by F.conv2d',
                     'input_weight_residency':'GPU except explicit host_to_host',
                     'host_buffers':'reused pinned input, weight and output; reusable device staging buffers',
                     'workspace_limit':'framework managed, no explicit byte cap' if name.startswith('torch') else 0}
                for batch in range(protocol['batches']):
                    conditions = [(res,cache) for res in calls for cache in ('steady','eviction_attempt')]
                    random.Random(f"{shape.id}:{name}:{batch}:{protocol['seed']}").shuffle(conditions)
                    for order,(residency,cache) in enumerate(conditions):
                        call=calls[residency]
                        for index in range(protocol['samples_per_batch']):
                            if residency=='host_to_host':
                                if cache=='eviction_attempt': evict(); stream.synchronize()
                                stream.synchronize(); begin=time.perf_counter_ns(); call(); stream.synchronize()
                                value={'t_host_to_host_us':(time.perf_counter_ns()-begin)/1000}
                            else: value=sample(call,torch,stream,evict if cache=='eviction_attempt' else None)
                            samples.append({'implementation':name,'batch':batch,'sample':index,'condition_order':order,
                                            'residency':residency,'cache_policy':cache,**value})
                rows.append(row)
        result={'shape':record,'rows':rows,'samples':samples,'conversions':conversions,'precision':settings,
                'gpu':identity_gpu(torch),'nvidia_smi':gpu_snapshot(),
                'layout_initialization_search_us':{'NCHW':reference_search,'channels_last':channels_last_search},
                'eviction':{'bytes':protocol['eviction_bytes'],'method':'FP32 fill on same stream; completion before timed interval',
                            'all_caches_cleared':'NOT_VERIFIED'},
                'order':'variant order fixed; residency/cache order independently seeded each batch',
                'search_cache':'one fresh process per shape; layout plans may reuse process-local framework cache'}
    except BaseException as exc:
        result={'shape':record,'rows':rows,'samples':samples,'status':'FAIL','reason':str(exc)}
    connection.send(result); connection.close()


def summarize(data):
    summaries=[]
    for row in data['rows']:
        for residency in ('target_layout_ready','caller_nchw','host_to_host'):
            for cache in ('steady','eviction_attempt'):
                selected=[s for s in data['samples'] if s['implementation']==row['implementation'] and s['residency']==residency and s['cache_policy']==cache]
                if not selected: continue
                stats={}
                for field in ('t_device_op_graph_us','t_api_us','t_host_to_host_us'):
                    if field not in selected[0]: continue
                    stats[field]=describe([s[field] for s in selected])
                    stats[field]['batch_medians']=[float(np.median([s[field] for s in selected if s['batch']==b])) for b in sorted({s['batch'] for s in selected})]
                summaries.append({'shape_id':data['shape']['shape_id'],'implementation':row['implementation'],
                                  'residency':residency,'cache_policy':cache,'sample_count':len(selected),**stats})
    return summaries


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--correctness',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/cuda-boundaries-protocol.json')
    p.add_argument('--shape-ids',nargs='+'); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv); protocol=json.loads(args.config.read_text())
    if args.smoke: protocol.update(batches=1,samples_per_batch=2,conversion_samples=2)
    shapes=[r for r in json.loads((ROOT/'configs/atlas.json').read_text())['shapes'] if r['shape_id'] in (args.shape_ids or protocol['shape_ids'])]
    if len(shapes)!=len(set(args.shape_ids or protocol['shape_ids'])): raise ValueError('shape selection incomplete')
    identity=source_identity(); native=Native()
    validate_correctness(json.loads(args.correctness.read_text()),identity,sha(native.path),args.smoke)
    out=args.output
    if out.exists(): raise ValueError('use a new output directory')
    out.mkdir(parents=True)
    atomic_json(out/'provenance.json',{'source':identity,'library_sha256':sha(native.path),'correctness_sha256':sha(args.correctness),
                                     'protocol':protocol,'measurement_kind':'SMOKE' if args.smoke else 'FORMAL'})
    context=mp.get_context('spawn'); results=[]; summaries=[]; process=None
    previous=signal.signal(signal.SIGTERM,terminate_run)
    try:
        for record in shapes:
            parent,child=context.Pipe(duplex=False)
            process=context.Process(target=worker,args=(child,record,protocol,native.path))
            data={'shape':record,'status':'TIMEOUT','rows':[],'samples':[]}
            try:
                process.start(); child.close()
                if parent.poll(protocol['shape_timeout_seconds']):
                    try: data=parent.recv()
                    except EOFError: data.update(status='FAIL',reason='shape worker exited without receipt')
            finally: kill_shape_group(process); parent.close(); child.close()
            atomic_json(out/'shapes'/f"{record['shape_id']}.json",data)
            results.append({'shape_id':record['shape_id'],'status':data.get('status','PASS'),'rows':len(data['rows'])})
            summaries.extend(summarize(data)); atomic_json(out/'summary.json',{'shapes':results,'measurements':summaries})
            print(json.dumps(results[-1]),flush=True)
    finally:
        kill_shape_group(process)
        signal.signal(signal.SIGTERM,previous)
    return int(any(r['status']!='PASS' for r in results))


if __name__=='__main__': raise SystemExit(main())
