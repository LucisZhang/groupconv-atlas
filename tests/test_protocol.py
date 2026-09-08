import json
from pathlib import Path
import numpy as np
import pytest
from bench.api import ROOT,Shape,Native,inputs,cshape
from bench.stats import compare_batches,describe

def test_frozen_shape_and_split_closure():
    course=json.loads((ROOT/'configs/course.json').read_text())['shapes']
    atlas=json.loads((ROOT/'configs/atlas.json').read_text())['shapes']
    for rows,count in ((course,36),(atlas,80)):
        assert len(rows)==count and len({r['shape_id'] for r in rows})==count
        for row in rows:
            s=Shape.from_record(row)
            assert row['shape_id']==s.id and row['output_dims']==list(s.output)
    split=json.loads((ROOT/'configs/split.json').read_text())
    tuning,holdout=set(split['tuning']),set(split['holdout'])
    assert not tuning&holdout and tuning|holdout=={r['shape_id'] for r in atlas}
    for part in (tuning,holdout):
        assert {r['cin']//r['groups'] for r in atlas if r['shape_id'] in part}=={1,2,4,8,16,32,64,256}

def test_confirmation_uses_independent_batches():
    assert compare_batches([2]*5,[1]*5)['classification']=='UNCERTAIN'
    assert compare_batches([2]*10,[1]*10)['classification']=='WIN'
    assert compare_batches([1]*10,[2]*10)['classification']=='LOSS'
    assert compare_batches([1]*10,[1]*10)['classification']=='TIE'
    for bad in (np.nan,np.inf,0,-1):
        with pytest.raises(ValueError): compare_batches([bad]*10,[1]*10)
    with pytest.raises(ValueError): describe([0])

def test_native_integer_wrap_rejected():
    with pytest.raises(ValueError): cshape(Shape(n=2**64+1))
    with pytest.raises(ValueError): cshape(Shape(n=1.5))

def test_cpu_benchmark_has_no_device_time():
    from bench.run import native_measure
    result=native_measure(lambda: None,3,1,10,2)
    assert len(result['api'])==3 and result['device']==[]
    assert all(t>0 for t in result['api'])
