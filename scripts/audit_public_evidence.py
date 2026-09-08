"""Validate public hashes and recompute the RTX 4090 atlas from raw samples (stdlib only)."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
SESSION=Path('results/rtx4090/session-20260908-vast-03')
FIELDS=('t_device_op_graph_us','t_api_us')
VARIANTS={'k0','k1','k2','k3','torch_cuda_default','torch_cuda_tuned'}
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def audit(root,require_license=False):
    root=Path(root).resolve();manifest=read(root/'manifest.json')
    require(manifest['schema']=='groupconv-public-evidence-v2','unknown public schema')
    if require_license: require(manifest['license_choice']!='PENDING_OWNER_SELECTION','license selection pending')
    seen=set()
    for entry in manifest['files']:
        name=entry['path'];p=(root/name).resolve()
        require(name not in seen and not Path(name).is_absolute() and p.is_relative_to(root),'duplicate/unsafe manifest path')
        seen.add(name);data=p.read_bytes()
        require(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],'payload hash mismatch: '+name)
        if entry.get('byte_identical'):require(entry['source_sha256']==entry['sha256'],'original hash differs for unchanged payload')
    expected={r['shape_id'] for r in read(root/'configs/atlas.json')['shapes']}
    require(len(expected)==80,'atlas configuration must contain 80 unique IDs')
    base=root/SESSION/'complete-evidence/results/atlas-confirmation/atlas'
    rows=read(base/'bench.json');keys={(r['shape_id'],r['implementation']) for r in rows}
    require(len(rows)==len(keys)==480 and keys=={(sid,v) for sid in expected for v in VARIANTS},'incomplete/duplicate bench matrix')
    require(Counter(r['status'] for r in rows)=={'PASS':430,'UNSUPPORTED':50},'unexpected result denominator')
    grouped=defaultdict(dict);sample_count=0
    with (base/'samples.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=r['shape_id'],r['implementation'];slot=r['batch'],r['sample']
            require(key in keys and type(slot[0]) is int and 0<=slot[0]<10 and type(slot[1]) is int and 0<=slot[1]<30,'sample identity outside plan')
            require(slot not in grouped[key],'duplicate raw sample')
            for field in FIELDS:require(type(r[field]) in (int,float) and math.isfinite(r[field]) and r[field]>0,'nonpositive/nonfinite time')
            grouped[key][slot]=r;sample_count+=1
    medians={}
    for r in rows:
        key=r['shape_id'],r['implementation']
        if r['status']=='UNSUPPORTED':
            require(r['implementation']=='k2' and key not in grouped,'unsupported result has samples')
            continue
        require(len(grouped[key])==r['sample_count']==300,'incomplete sample grid')
        require(r['error']['passed'] is True and r['error']['failed_elements']==0,'FP32 correctness gate failed')
        require(r['fp64_points']['passed'] is True and r['fp64_points']['point_count']==64 and r['fp64_points']['failed_points']==0,'FP64 point gate failed')
        for field in FIELDS:
            medians[key+(field,)]=[statistics.median(grouped[key][b,s][field] for s in range(30)) for b in range(10)]
    require(sample_count==129000 and len(grouped)==430,'raw denominator mismatch')
    analysis=read(root/SESSION/'analysis-atlas/analysis.json');ratios=defaultdict(list);classifications=defaultdict(Counter)
    for row in analysis['comparisons']:
        key=row['baseline'],row['candidate'],row['boundary'];classifications[key][row['classification']]+=1
        if row['classification']=='NOT_COMPARABLE':continue
        sid=row['shape_id'];left=medians[sid,row['baseline'],row['boundary']];right=medians[sid,row['candidate'],row['boundary']]
        require(left==row['baseline_batch_medians'] and right==row['candidate_batch_medians'],'analysis batch medians differ from raw')
        ratio=statistics.median(left)/statistics.median(right)
        require(math.isclose(ratio,row['ratio'],rel_tol=1e-12),'reported ratio differs from raw')
        lo,hi=row['ci95'];threshold=1.05
        classification='WIN' if lo>threshold else 'LOSS' if hi<1/threshold else 'TIE' if lo>=1/threshold and hi<=threshold else 'UNCERTAIN'
        require(classification==row['classification'],'CI classification inconsistent')
        ratios[key].append(ratio)
    results=[]
    for row in analysis['aggregates']:
        key=row['baseline'],row['candidate'],row['boundary'];values=ratios[key]
        require(len(values)==row['comparable'] and dict(classifications[key])==row['classification_counts'],'aggregate denominator/classification mismatch')
        value=math.exp(statistics.mean(math.log(v) for v in values))
        require(math.isclose(value,row['geomean'],rel_tol=1e-12),'geomean differs from raw')
        if key[1]=='k3' and key[0] in ('k0','torch_cuda_tuned'):results.append({'baseline':key[0],'candidate':key[1],'boundary':key[2],'geomean':value})
    identity=read(root/'evidence/measured-source-identity.json');files=identity['source']['source_files']
    require(hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()==identity['source']['source_tree_sha256'],'measured source identity invalid')
    require(set(identity['published_files']).isdisjoint(identity['omitted_files']) and set(identity['published_files'])|set(identity['omitted_files'])==set(files),'source subset not closed')
    for name in identity['published_files']:
        p=root/'evidence/measured-source-031887c'/name
        require(hashlib.sha256(p.read_bytes()).hexdigest()==files[name],'measured source differs: '+name)
    return {'status':'PASS','payload_files':len(seen),'shapes':80,'pass_rows':430,'unsupported_rows':50,'samples':sample_count,'ratios':results,
            'scope':'Public payload/hash, original-source subset, raw denominator/point statistics and reported CI classification. Bootstrap endpoints and private execution proof are separate audits.'}
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--require-license',action='store_true')
    a=p.parse_args(argv);print(json.dumps(audit(a.root,a.require_license),indent=2))
if __name__=='__main__':main()
