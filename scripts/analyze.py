"""Derive course tables/charts directly from raw batch records."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from bench.stats import compare_batches

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(a.run/'plots'/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((a.run/'bench.json').read_text()); good=[r for r in rows if r['status']=='PASS']
    keyed={(r['shape_id'],r['implementation'],r['threads']):r for r in good}
    shapes=json.loads((a.run/'shapes.json').read_text()); stats=[]; scaling=[]
    for s in shapes:
        sid=s['shape_id']
        for impl in ('c1','c2'):
            if (sid,impl,1) not in keyed: continue
            b=keyed[(sid,impl,1)]
            for threads in (1,2,4,8):
                if (sid,impl,threads) not in keyed: continue
                r=keyed[(sid,impl,threads)]
                result=compare_batches(b['batch_medians_api_us'],r['batch_medians_api_us'])
                speedup=b['t_api_us']/r['t_api_us']
                scaling.append({'shape_id':sid,'implementation':impl,'threads':threads,
                                'speedup':speedup,'efficiency':speedup/threads,
                                'classification':result['classification'],'cpg':s['cin']//s['groups']})
        for candidate in ('o1','o2'):
            b=keyed.get((sid,'o0',1)); r=keyed.get((sid,candidate,1))
            if not b or not r: continue
            for boundary in ('device_op','api','host_to_host'):
                result=compare_batches(b[f'batch_medians_{boundary}_us'],r[f'batch_medians_{boundary}_us'])
                result['batch_median_speedup']=result['speedup']
                result['speedup']=b[f't_{boundary}_us']/r[f't_{boundary}_us']
                result['point_estimate']='ratio of medians of all raw samples; bootstrap uses paired batch medians'
                stats.append({'shape_id':sid,'candidate':candidate,'baseline':'o0','boundary':boundary,**result})
    (a.run/'comparisons.json').write_text(json.dumps(stats,indent=2)+'\n')
    (a.run/'scaling.json').write_text(json.dumps(scaling,indent=2)+'\n')
    with (a.run/'scaling.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,list(scaling[0]) if scaling else ['shape_id']); wr.writeheader(); wr.writerows(scaling)
    dest=a.run/'plots'; dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':11})
    fig,ax=plt.subplots(figsize=(7.6,4.5))
    for impl,label in (('c1','OpenMP outputs'),('c2','OpenMP spatial blocking')):
        means=[]
        threads=sorted({r['threads'] for r in scaling if r['implementation']==impl})
        for t in threads:
            values=[r['speedup'] for r in scaling if r['implementation']==impl and r['threads']==t]
            means.append(float(np.exp(np.mean(np.log(values)))))
        ax.plot(threads,means,marker='o',label=label)
    ax.plot([1,8],[1,8],'--',color='.7',label='Ideal linear scaling')
    ax.set(xlabel='Threads',ylabel='Speedup vs same algorithm at 1 thread',title='CPU scaling: geometric mean over 36 course shapes')
    ax.set_xticks([1,2,4,8]); ax.legend(); fig.tight_layout(); fig.savefig(dest/'cpu-scaling.png',dpi=180); plt.close(fig)
    if stats:
        data=np.full((9,4),np.nan)
        labels=[]
        for i,(c,h) in enumerate(( (c,h) for c in (16,32,64) for h in (16,32,64))):
            labels.append(f'C={c}, H=W={h}')
            for j,g in enumerate((1,2,4,c)):
                ss=next((s for s in shapes if s['cin']==c and s['h']==h and s['groups']==g),None)
                if ss:
                    values=[v['speedup'] for v in stats if v['shape_id']==ss['shape_id'] and v['candidate']=='o2' and v['boundary']=='device_op']
                    if values: data[i,j]=values[0]
        fig,ax=plt.subplots(figsize=(7.5,6.5))
        limit=max(.1,float(np.nanmax(np.abs(np.log2(data)))))
        im=ax.imshow(np.log2(data),cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(4),['G=1','G=2','G=4','Depthwise']); ax.set_yticks(range(9),labels)
        for i in range(9):
            for j in range(4):
                if np.isfinite(data[i,j]): ax.text(j,i,f'{data[i,j]:.2f}x',ha='center',va='center',fontsize=11,
                    color='white' if abs(np.log2(data[i,j]))>limit*.62 else 'black')
        ax.set_title('Naive time / register-reuse time\nDevice events; descriptive raw-sample medians')
        fig.colorbar(im,ax=ax,label='log2 speedup'); fig.tight_layout(); fig.savefig(dest/'gpu-register-map.png',dpi=180); plt.close(fig)
    # Manifest generated from actual files, never names alone.
    manifest=[]
    for path in sorted(a.run.rglob('*')):
        if path.is_file() and '.mpl-cache' not in path.parts and path.name!='manifest.json':
            manifest.append({'path':str(path.relative_to(a.run)),'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'category':'raw_samples' if path.name=='samples.jsonl' else 'derived_or_provenance',
                'generated_by':'bench.run or scripts.analyze','license':'project-authored data; no third-party source code included'})
    (a.run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'comparison_rows':len(stats),'scaling_rows':len(scaling),'manifest_files':len(manifest)}))
if __name__=='__main__': main()
