"""Batch-level statistics. Adjacent samples are never independent replicates."""
import numpy as np

def describe(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.all(np.isfinite(a)) or np.any(a<=0):
        raise ValueError('timings must be finite and positive')
    return {'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),
            'p90':float(np.quantile(a,.9))}

def compare_batches(base,candidate,minimum=10,seed=20260908,draws=10000):
    base=np.asarray(base,dtype=float); candidate=np.asarray(candidate,dtype=float)
    if base.ndim!=1 or candidate.ndim!=1 or len(base)!=len(candidate) or not len(base) or not np.all(np.isfinite(base)) or not np.all(np.isfinite(candidate)) or np.any(base<=0) or np.any(candidate<=0):
        raise ValueError('paired positive batch medians required')
    value=float(np.median(base)/np.median(candidate))
    if len(base)<minimum:
        return {'speedup':value,'classification':'UNCERTAIN','ci95':None,
                'paired_batches':len(base),'reason':f'confirmation requires {minimum} independent batches'}
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(base),size=(draws,len(base)))
    ratios=np.median(base[idx],axis=1)/np.median(candidate[idx],axis=1)
    low,high=map(float,np.quantile(ratios,[.025,.975])); threshold=1.05
    classification=('WIN' if low>threshold else 'LOSS' if high<1/threshold
                    else 'TIE' if low>=1/threshold and high<=threshold else 'UNCERTAIN')
    return {'speedup':value,'classification':classification,'ci95':[low,high],
            'paired_batches':len(base),'method':'paired batch median block bootstrap','draws':draws}
