"""Archive committed measurement sources with exact file hashes and no host access data."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from bench.api import ROOT
from bench.run import source_identity, atomic_json, sha

def create(output):
    identity=source_identity(); files=identity['source_files']
    if not identity['source_commit']: raise RuntimeError('commit the source first')
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    if set(files)-tracked: raise RuntimeError('source contains untracked files; commit them before bundling')
    subprocess.run(['git','diff','--exit-code','HEAD','--',*files],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    output=Path(output)
    if output.exists() or output.with_suffix('.json').exists(): raise ValueError('bundle output must be new')
    output.parent.mkdir(parents=True,exist_ok=True)
    data=io.BytesIO()
    with gzip.GzipFile(fileobj=data,mode='wb',mtime=0,filename='') as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for name in sorted(files):
                path=ROOT/name
                if path.is_symlink(): raise ValueError('measurement source symlink rejected')
                raw=path.read_bytes(); entry=tarfile.TarInfo(name)
                entry.size=len(raw);entry.mode=0o755 if name=='run.sh' else 0o644;entry.mtime=0
                archive.addfile(entry,io.BytesIO(raw))
    output.write_bytes(data.getvalue())
    manifest={'kind':'committed_measurement_source_bundle','source':identity,
              'archive':output.name,'bytes':output.stat().st_size,'sha256':sha(output),
              'files':{name:{'sha256':digest,'bytes':(ROOT/name).stat().st_size} for name,digest in files.items()}}
    with tarfile.open(output,'r:gz') as archive:
        members={m.name:m for m in archive.getmembers()}
        if set(members)!=set(files): raise RuntimeError('archive coverage mismatch')
        for name,digest in files.items():
            if hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest()!=digest:
                raise RuntimeError('archive file hash mismatch')
    manifest['verification']='PASS';atomic_json(output.with_suffix('.json'),manifest)
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);result=create(args.output)
    print(json.dumps({k:result[k] for k in ('archive','bytes','sha256','verification')}))

if __name__=='__main__': main()
