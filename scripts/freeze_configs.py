"""Explicit configuration generation. Benchmarks only read these files."""
import json
from pathlib import Path
from bench.api import Shape

ROOT=Path(__file__).resolve().parents[1]
def save(name,value):
    (ROOT/'configs'/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def main():
    course=[Shape(cin=c,cout=c,h=h,w=h,groups=g).record()
            for c in (16,32,64) for h in (16,32,64) for g in (1,2,4,c)]
    atlas=[Shape(n=n,cin=256,cout=256,h=h,w=h,groups=256//cpg).record()
           for n in (1,32) for h in (7,14,28,56,112) for cpg in (1,2,4,8,16,32,64,256)]
    tuning=[]; holdout=[]
    for cpg in (1,2,4,8,16,32,64,256):
        ids=sorted(s['shape_id'] for s in atlas if s['cin']//s['groups']==cpg)
        tuning.extend(ids[:5]); holdout.extend(ids[5:])
    save('course.json',{'schema_version':1,'frozen_date':'2026-09-08','count':36,'shapes':course})
    batch4=[Shape(n=4,cin=16,cout=16,h=16,w=16,groups=g).record() for g in (1,4,16)]
    save('course-batch4.json',{'schema_version':1,'frozen_date':'2026-09-08','count':3,'shapes':batch4})
    save('atlas.json',{'schema_version':1,'frozen_date':'2026-09-08','count':80,'shapes':atlas})
    save('split.json',{'rule':'first five lexicographic shape hashes per cpg tune; remaining five holdout',
                       'tuning':sorted(tuning),'holdout':sorted(holdout),'dispatcher_status':'NOT_IMPLEMENTED'})
    save('protocol.json',{
        'schema_version':1,'frozen_date':'2026-09-08','seed':20260908,'correctness_seeds':[20260908,17,101],
        'dtype':'float32','accumulation':'float32','tf32':False,'bias':False,'layout':'NCHW',
        'atol':1e-4,'rtol':1e-4,'relative_epsilon':1e-12,'finite_required':True,
        'warmup':10,'batches':5,'samples_per_batch':30,'confirmation_batches':10,
        'threads':[1,2,4,8],'repeat_target_us':1000,'repeat_cap':100,
        'cache_policy':'steady_state_repeated_inputs','eviction_status':'NOT_RUN',
        'order':'seeded shuffle implementation blocks within each paired batch',
        'api_boundary':'synchronized host wall clock; borrowed/reused native buffers; torch allocates output',
        'opencl_event_boundary':'per-logical-op START to END event duration; repeated calls averaged',
        'host_to_host_boundary':'upload input+weights, logical convolution, download output; reusable device buffers',
        'layout_conversions':'none for NCHW stage B','search_cost':'no autotune; fixed candidates',
        'confidence':{'threshold':1.05,'method':'paired batch median block bootstrap','bootstrap_draws':10000,
                      'confidence_level':0.95,'minimum_batches':10},
        'budget_cny':{'total_cap':300,'probe_cap':30,'cuda_core_cap':120,'profiling_cap':80,'storage_cap':30,'reserve_cap':40},
        'stage_C':'PLANNED','stage_D':'PLANNED'})
    print('Frozen 36 course shapes, 80 atlas shapes, disjoint 40/40 tuning/holdout')

if __name__=='__main__': main()
