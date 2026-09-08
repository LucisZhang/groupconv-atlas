"""Fixed MobileNetV2 block substitution and instrumented Amdahl probe; random weights, no accuracy claim."""
from __future__ import annotations
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def discover(model, torch):
    block = model.features[3]
    found = [(name, layer) for name, layer in block.named_modules()
             if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size == (3,3)
             and layer.stride == (1,1) and layer.padding == (1,1)
             and layer.dilation == (1,1) and layer.groups == layer.in_channels == layer.out_channels
             and layer.bias is None and layer.padding_mode == 'zeros']
    if len(found) != 1: raise RuntimeError(f'expected exactly one eligible depthwise child; found {len(found)}')
    return block, found[0][0], found[0][1]


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_record(value):
    return {'shape':list(value.shape), 'stride':list(value.stride()), 'dtype':str(value.dtype)}


def hook_trace(model, block, conv, tx, stream=None):
    observed = {}
    def capture(name):
        def hook(layer, inputs, output):
            observed[name] = {'input':inputs[0].detach().clone(), 'output':output.detach().clone(),
                              'input_meta':tensor_record(inputs[0]), 'output_meta':tensor_record(output)}
        return hook
    handles = [block.register_forward_hook(capture('block')), conv.register_forward_hook(capture('conv'))]
    try:
        result = model(tx)
        if stream is not None: stream.synchronize()
    finally:
        for handle in handles: handle.remove()
    return observed, result


def stop(signum, frame):
    raise TimeoutError(f'probe interrupted by signal {signum}')


def profile_blocks(torch, stream, baseline_call, candidate_call, conv, replacement, output):
    """One post-measurement eager trace. Trace durations never enter benchmark samples."""
    result={'status':'RUNNING','excluded_from_timing_samples':True,
            'target_device_time_fraction_p':'UNKNOWN',
            'p_reason':'CPU module markers alone do not establish exact ownership of asynchronous GPU work; no Amdahl estimate',
            'scope':'one eager original block and one eager substituted block, after all timed batches'}
    contexts={};handles=[]
    def pre(label):
        def before(module, inputs):
            context=torch.profiler.record_function(label);context.__enter__();contexts[module]=context
        return before
    def after(module, inputs, value):
        context=contexts.pop(module,None)
        if context is not None:context.__exit__(None,None,None)
    try:
        for module,label in ((conv,'module_probe::target_conv::torch'),(replacement,'module_probe::target_conv::k3')):
            handles.append(module.register_forward_pre_hook(pre(label)))
            handles.append(module.register_forward_hook(after))
        stream.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,profile_memory=False,with_stack=False) as profiler:
            with torch.profiler.record_function('module_probe::block::torch_tuned'):
                baseline_call();stream.synchronize()
            with torch.profiler.record_function('module_probe::block::one_k3_substitution'):
                candidate_call();stream.synchronize()
        path=output/'block-profile-trace.json'
        profiler.export_chrome_trace(str(path))
        trace=json.loads(path.read_text());events=trace.get('traceEvents',[])
        kernels=[{'name':e.get('name'),'category':e.get('cat'),'duration_us':e.get('dur'),
                  'timestamp_us':e.get('ts'),'pid':e.get('pid'),'tid':e.get('tid'),'args':e.get('args',{})}
                 for e in events if 'kernel' in str(e.get('cat','')).lower() and e.get('ph')=='X']
        annotations=[{'name':e.get('name'),'category':e.get('cat'),'duration_us':e.get('dur'),
                      'timestamp_us':e.get('ts'),'pid':e.get('pid'),'tid':e.get('tid')}
                     for e in events if str(e.get('name','')).startswith('module_probe::')]
        grouped={}
        for event in kernels:
            name=event['name'];row=grouped.setdefault(name,{'name':name,'count':0,'summed_trace_duration_us':0.0})
            row['count']+=1;row['summed_trace_duration_us']+=float(event['duration_us'] or 0)
        categories={}
        for event in events:
            category=event.get('cat','UNSPECIFIED');categories[category]=categories.get(category,0)+1
        evidence={'kernel_events':kernels,'kernel_name_summary':list(grouped.values()),'module_markers':annotations,
                  'event_categories':categories,'target_device_time_fraction_p':'UNKNOWN',
                  'interpretation':'profiler trace evidence only; sums include instrumented execution and do not replace latency samples'}
        atomic(output/'block-profile-events.json',evidence)
        result.update(status='PASS' if kernels else 'NOT_RUN',trace_path=str(path),trace_sha256=digest(path),
                      event_summary_path=str(output/'block-profile-events.json'),kernel_event_count=len(kernels),
                      module_marker_count=len(annotations))
        if not kernels:result['reason']='trace contains no CUDA kernel events'
    except BaseException as exc:
        result.update(status='TIMEOUT' if isinstance(exc,TimeoutError) else 'FAIL',reason=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            for handle in handles:handle.remove()
            for context in contexts.values():context.__exit__(None,None,None)
        except BaseException as exc:
            result.update(status='FAIL',cleanup_reason=f'{type(exc).__name__}: {exc}')
    return result


def amdahl_probe(torch, stream, baseline_call, conv, candidate_call, replacement, output):
    """Post-benchmark graph event decomposition; instrumentation stays out of main samples."""
    import numpy as np
    data={}
    for name,call,leaf in (('baseline',baseline_call,conv),('candidate',candidate_call,replacement)):
        events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(4)]
        handles=[leaf.register_forward_pre_hook(lambda *args: events[1].record(stream)),
                 leaf.register_forward_hook(lambda *args: events[2].record(stream))]
        try:
            graph=torch.cuda.CUDAGraph();stream.synchronize()
            with torch.cuda.graph(graph,stream=stream):
                events[0].record(stream);result=call();events[3].record(stream)
        finally:
            for handle in handles: handle.remove()
        graph.replay();stream.synchronize()
        rows=[]
        for index in range(30):
            graph.replay();stream.synchronize()
            whole=events[0].elapsed_time(events[3])*1000
            target=events[1].elapsed_time(events[2])*1000
            rows.append({'sample':index,'block_graph_us':whole,'target_graph_us':target,'p':target/whole})
        data[name]={'samples':rows,'block_graph_us':float(np.median([r['block_graph_us'] for r in rows])),
                    'target_graph_us':float(np.median([r['target_graph_us'] for r in rows]))}
    p=data['baseline']['target_graph_us']/data['baseline']['block_graph_us']
    s=data['baseline']['target_graph_us']/data['candidate']['target_graph_us']
    valid=0<=p<=1 and s>0
    receipt={'status':'PASS' if valid else 'INVALID_FRACTION','measurements':data,'p':p,'s':s,
             'predicted_block_speedup':1/((1-p)+p/s) if valid else None,
             'observed_instrumented_block_speedup':data['baseline']['block_graph_us']/data['candidate']['block_graph_us'],
             'boundary':'four external event nodes in one captured complete block, target bracketed by hooks; one same non-default stream',
             'excluded_from_main_latency_samples':True,
             'limitations':'instrumented estimate; graph event nodes may perturb scheduling; assumes other block work unchanged; not eager API or whole-model prediction'}
    atomic(output/'amdahl-instrumented.json',receipt)
    return receipt


def run(args, report):
    sys.path.insert(0, str(args.root.resolve()))
    import numpy as np
    import torch
    import torchvision
    from bench.api import Shape, Native, error_metrics
    from bench.cuda_run import precision, timing, identity_gpu, gpu_snapshot
    from bench.run import source_identity
    from bench.stats import describe
    from bench.fp64_points import prepare_reference, check_points
    report.update(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), source=source_identity(),
                  torch_git=torch.version.git_version, script_sha256=digest(__file__))
    if str(torchvision.__version__).split('+')[0] != '0.23.0': raise RuntimeError('formal module probe requires torchvision 0.23.0')
    torch.manual_seed(args.seed)
    model = torchvision.models.mobilenet_v2(weights=None).eval()
    block, relative_path, conv = discover(model, torch)
    source_paths = {inspect.getsourcefile(type(model)), inspect.getsourcefile(type(block)), inspect.getsourcefile(type(conv))}
    report.update(model='torchvision.models.mobilenet_v2', weights=None, random_seed=args.seed,
                  state_sha256=state_digest(model), block_path='features.3', selected_layer_path='features.3.'+relative_path,
                  selected_layer_repr=repr(conv), block_repr=repr(block), fixed_implementation='k3',
                  selection_rule='features[3], unique instantiated depthwise Conv2d 3x3 stride1 padding1; k3 fixed before measurements',
                  module_source_files={p:digest(p) for p in sorted(p for p in source_paths if p)})
    # Preserve the exact random state as well as its content hash; no downloads occur.
    torch.save(model.state_dict(), args.output/'random-state.pt')
    report['state_file_sha256'] = digest(args.output/'random-state.pt')
    if not torch.cuda.is_available():
        report.update(status='NOT_RUN', reason='real CUDA unavailable; structural discovery only')
        return 2
    settings = precision(torch)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, 'benchmark_limit'): torch.backends.cudnn.benchmark_limit = 10
    native = Native(); report.update(library_sha256=digest(native.path), native_build=native.lib.gc_build_info().decode(),
                                    precision=settings, gpu=identity_gpu(torch), nvidia_smi=gpu_snapshot())
    if not native.lib.gc_cuda_available(): raise RuntimeError('native CUDA unavailable')
    model = model.cuda(); candidate = copy.deepcopy(model).eval()
    block, relative_path, conv = discover(model, torch)
    stream = torch.cuda.Stream(device=0)
    generator = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    tx = torch.randn((args.batch_size,3,224,224), generator=generator).cuda()
    report['input_sha256'] = hashlib.sha256(tx.cpu().numpy().tobytes()).hexdigest()
    with torch.inference_mode(), torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        begin=time.perf_counter_ns()
        observed, model_reference = hook_trace(model, block, conv, tx, stream)
        report['model_first_call_with_hooks_us']=(time.perf_counter_ns()-begin)/1000
        report['model_first_call_with_hooks_boundary']='includes initialization/search, forward hooks, tensor clones, metadata recording and synchronization; not pure search cost'
        report['observed']={name:{'input':v['input_meta'],'output':v['output_meta']} for name,v in observed.items()}
        block_input = observed['block']['input']; conv_input = observed['conv']['input']
        class NativeDepthwise(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('weight', conv.weight.detach().clone())
                self.last_stream = None; self.last_input_copy = False
            def forward(self, x):
                expected=(x.shape[1]*x.shape[2]*x.shape[3],x.shape[2]*x.shape[3],x.shape[3],1)
                self.last_input_copy = tuple(x.stride()) != expected
                if self.last_input_copy:
                    packed=torch.empty(tuple(x.shape),dtype=x.dtype,device=x.device)
                    packed.copy_(x); x=packed
                shape=Shape(n=x.shape[0],cin=conv.in_channels,cout=conv.out_channels,h=x.shape[2],w=x.shape[3],groups=conv.groups)
                output=torch.empty(shape.output,dtype=x.dtype,device=x.device)
                self.last_stream=int(torch.cuda.current_stream(x.device).cuda_stream)
                native.cuda(shape,x,self.weight,output,3,self.last_stream)
                return output
        replacement = NativeDepthwise().eval()
        candidate_block=candidate.features[3]
        parts=relative_path.split('.'); parent=candidate_block
        for part in parts[:-1]: parent=parent._modules[part]
        parent._modules[parts[-1]]=replacement
        def metrics(actual, expected):
            stream.synchronize()
            return error_metrics(actual.cpu().numpy(),expected.cpu().numpy())
        checks={}
        checks['leaf_conv']=metrics(replacement(conv_input),observed['conv']['output'])
        checks['complete_block']=metrics(candidate_block(block_input),observed['block']['output'])
        checks['complete_model']=metrics(candidate(tx),model_reference)
        shape=Shape(n=conv_input.shape[0],cin=conv.in_channels,cout=conv.out_channels,h=conv_input.shape[2],w=conv_input.shape[3],groups=conv.groups)
        prepared=prepare_reference(shape,conv_input.cpu().numpy(),conv.weight.detach().cpu().numpy(),seed=args.seed,count=64)
        checks['leaf_fp64_points']=check_points(prepared,replacement(conv_input).cpu().numpy())
        report['selected_shape']=shape.record()
        report.update(checks=checks, native_stream={'pointer':replacement.last_stream,'non_default':replacement.last_stream!=0,
                      'selection':'torch.cuda.current_stream(input.device) each call'},
                      native_input_materialization=bool(replacement.last_input_copy))
        atomic(args.output/'summary.json',report)
        if not all(value['passed'] for value in checks.values()):
            report.update(status='FAIL',reason='full-output leaf/block/model comparison failed')
            return 1
        def baseline_call(): return block(block_input)
        def candidate_call(): return candidate_block(block_input)
        calls={'torch_block_tuned':baseline_call,'block_one_k3_substitution':candidate_call}
        repeats={}; status={}; samples=[]
        for name,call in calls.items():
            try:
                begin=time.perf_counter_ns()
                for _ in range(10): call()
                stream.synchronize()
                warm=(time.perf_counter_ns()-begin)/1000
                pilot=timing(call,torch,stream,1)
                repeats[name]=max(1,min(20,int(1000/max(1,pilot['t_api_us']))))
                timing(call,torch,stream,repeats[name])
                # timing() itself verifies graph outputs against eager before sampling.
                graph,pairs,outputs=call._graph_cache[repeats[name]]
                graph.replay();stream.synchronize(); captured=outputs[-1].clone();stream.synchronize()
                graph_check=metrics(captured,call())
                status[name]={'status':'PASS' if graph_check['passed'] else 'FAIL',
                              'graph_vs_eager':graph_check,'warmup_total_us':warm,'repeats':repeats[name]}
            except Exception as exc:
                if isinstance(exc,TimeoutError): raise
                status[name]={'status':'OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAIL','reason':str(exc)}
        with (args.output/'samples.jsonl').open('w') as raw:
            for batch in range(args.batches):
                order=list(calls);random.Random(f'{args.seed}:{batch}').shuffle(order)
                for order_index,name in enumerate(order):
                    if status[name]['status']!='PASS':continue
                    try:
                        for sample in range(args.samples):
                            value={'implementation':name,'batch':batch,'sample':sample,'order':order_index,
                                   'repeats':repeats[name],**timing(calls[name],torch,stream,repeats[name])}
                            samples.append(value);raw.write(json.dumps(value)+'\n');raw.flush()
                    except Exception as exc:
                        if isinstance(exc,TimeoutError): raise
                        status[name].update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAIL',reason=str(exc))
                report.update(measurements=status,sample_count=len(samples),completed_batches=batch+1)
                atomic(args.output/'summary.json',report)
        for name in calls:
            values=[v for v in samples if v['implementation']==name]
            status[name]['sample_count']=len(values)
            for field in ('t_api_us','t_device_op_graph_us','t_host_feed_us'):
                if values:
                    stats=describe([v[field] for v in values])
                    meds=[float(np.median([v[field] for v in values if v['batch']==b])) for b in sorted({v['batch'] for v in values})]
                    stats.update(batch_medians=meds,batch_cv=float(np.std(meds,ddof=1)/np.mean(meds)) if len(meds)>1 else None)
                    status[name][field]=stats
        report.update(status='PASS' if all(v['status']=='PASS' for v in status.values()) else 'FAIL',measurements=status)
        if report['status']=='PASS':
            report['amdahl_instrumented']=amdahl_probe(torch,stream,baseline_call,conv,candidate_call,replacement,args.output)
            report['descriptive_block_speedup']={field:float(np.median(status['torch_block_tuned'][field]['batch_medians']))/float(np.median(status['block_one_k3_substitution'][field]['batch_medians']))
                                                for field in ('t_api_us','t_device_op_graph_us')}
        # Save validated numerical/timing outcomes before separately bounded profiling.
        atomic(args.output/'summary.json',report)
        previous_remaining=signal.alarm(0) if hasattr(signal,'alarm') else 0
        if hasattr(signal,'alarm'):signal.alarm(min(previous_remaining or 45,45))
        try:
            report['profile']=profile_blocks(torch,stream,baseline_call,candidate_call,conv,replacement,args.output)
        finally:
            if hasattr(signal,'alarm'):signal.alarm(0)
        return 0 if report['status']=='PASS' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20260908)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--batches',type=int,default=5);parser.add_argument('--samples',type=int,default=30)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--timeout',type=int,default=600)
    args=parser.parse_args()
    if min(args.batches,args.samples,args.timeout,args.batch_size)<1:parser.error('positive counts and timeout required')
    if not args.smoke and (args.batches<5 or args.samples<30):parser.error('reduced runs require --smoke')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():parser.error('output already contains a run; use a new directory')
    report={'kind':'fixed_module_substitution_probe','status':'RUNNING','measurement_kind':'SMOKE' if args.smoke else 'DESCRIPTIVE',
            'batch_size':args.batch_size,'batches':args.batches,'samples_per_batch':args.samples,'benchmark':True,'deterministic':False,
            'timing_scope':'complete features[3] block, one fixed convolution replaced; all other operations retained',
            'eager_boundary':'synchronized API, including output allocation and required input materialization',
            'device_boundary':'graph-internal events around complete block; separate graph condition',
            'limitations':['random untrained weights; no model accuracy measurement','no whole-model performance claim',
                           'five-batch default is descriptive; no confidence win classification','steady-state fixed inputs; graph allocations retained',
                           'framework workspace managed; no explicit byte cap','actual backend dispatch UNKNOWN']}
    previous_term=signal.signal(signal.SIGTERM,stop)
    previous_alarm=signal.signal(signal.SIGALRM,stop) if hasattr(signal,'SIGALRM') else None
    if hasattr(signal,'alarm'):signal.alarm(args.timeout)
    started=time.monotonic()
    try:return_code=run(args,report)
    except BaseException as exc:
        report.update(status='TIMEOUT' if isinstance(exc,TimeoutError) else 'OOM' if 'OutOfMemory' in type(exc).__name__ else 'FAIL',reason=f'{type(exc).__name__}: {exc}')
        return_code=1
    finally:
        if hasattr(signal,'alarm'):signal.alarm(0);signal.signal(signal.SIGALRM,previous_alarm)
        signal.signal(signal.SIGTERM,previous_term)
        report['elapsed_seconds']=time.monotonic()-started
        atomic(args.output/'summary.json',report)
        source=Path(__file__).read_bytes();(args.output/'module_probe.py').write_bytes(source)
    print(json.dumps({'output':str(args.output),'status':report['status']}),flush=True)
    return return_code


if __name__=='__main__':raise SystemExit(main())
