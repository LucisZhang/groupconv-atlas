"""Independent scalar FP64 cross-correlation evidence at deterministic points.

CPU arrays only: CUDA owners must synchronize and explicitly copy to CPU first.
Sampled evidence is not a full-output correctness claim.
"""
from __future__ import annotations
import math
import operator
import numpy as np
from .api import Shape, FIELDS


def _shape(shape):
    shape = Shape.from_record(shape) if isinstance(shape, dict) else shape
    for field in FIELDS:
        value = operator.index(getattr(shape, field))
        if value < (0 if field in ('ph', 'pw') else 1):
            raise ValueError(f'invalid shape field: {field}')
    if shape.cin % shape.groups or shape.cout % shape.groups or min(shape.output) <= 0:
        raise ValueError('invalid groups or output geometry')
    return shape


def _cpu(value, dims):
    if hasattr(value, 'detach'):
        if value.device.type != 'cpu':
            raise ValueError('explicitly synchronize and copy tensor to CPU before oracle')
        value = value.detach()
        if str(value.dtype) == 'torch.bfloat16':
            value = value.to(dtype=__import__('torch').float64)
        value = value.numpy()
    result = np.asarray(value)
    if result.shape != tuple(dims):
        raise ValueError(f'array shape {result.shape} != expected {dims}')
    return result


def select_points(shape, count=64, seed=20260908):
    """Unique points: corners/middle/group boundaries first, seeded fill next.

Point budget may truncate the boundary candidates; receipts expose every point.
Returns all outputs when count is at least the output element count.
"""
    shape = _shape(shape)
    count = operator.index(count)
    if count <= 0:
        raise ValueError('count must be positive')
    dims = shape.output
    target = min(count, math.prod(dims))
    points, seen = [], set()
    def add(point):
        if len(points) < target and point not in seen:
            seen.add(point); points.append(point)
    n, c, h, w = dims
    # Cover batch endpoints, all four spatial corners and channel endpoints first.
    for batch in dict.fromkeys((0, n - 1)):
        for channel in dict.fromkeys((0, c - 1)):
            for row, col in ((0, 0), (h - 1, w - 1), (0, w - 1), (h - 1, 0), (h // 2, w // 2)):
                add((batch, channel, row, col))
    opg = c // shape.groups
    for group in range(shape.groups):
        for channel in dict.fromkeys((group * opg, (group + 1) * opg - 1)):
            add((group % n, channel, h // 2, w // 2))
        if len(points) == target:
            break
    rng = np.random.default_rng(seed)
    # Sample sparse budgets without allocating an output-sized permutation.
    while len(points) < target and target < math.prod(dims) // 2:
        add(tuple(int(rng.integers(d)) for d in dims))
    if len(points) < target:
        for point in np.ndindex(dims):
            add(point)
            if len(points) == target:
                break
    return points


def evaluate_points(shape, x, weights, actual, *, points=None, count=64,
                    seed=20260908, atol=1e-4, rtol=1e-4):
    """Return JSON-safe per-point positions/reference/actual/error/pass evidence.

x and weights are borrowed CPU NumPy/list/torch arrays. No torch/native conv is
called; reduction order is channel, kernel row, kernel column in Python FP64.
"""
    shape = _shape(shape)
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError('tolerances must be finite and nonnegative')
    x, weights, actual = (_cpu(x, shape.input), _cpu(weights, shape.weight), _cpu(actual, shape.output))
    selected = select_points(shape, count, seed) if points is None else [tuple(operator.index(v) for v in p) for p in points]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('points must be nonempty and unique')
    rows = []
    cpg, opg = shape.cin // shape.groups, shape.cout // shape.groups
    for point in selected:
        if len(point) != 4 or any(v < 0 or v >= d for v, d in zip(point, shape.output)):
            raise ValueError(f'output point out of bounds: {point}')
        n, co, oh, ow = point
        expected = 0.0
        for ci in range(cpg):
            for r in range(shape.r):
                ih = oh * shape.sh - shape.ph + r * shape.dh
                for s in range(shape.s):
                    iw = ow * shape.sw - shape.pw + s * shape.dw
                    if 0 <= ih < shape.h and 0 <= iw < shape.w:
                        expected += float(x[n, (co // opg) * cpg + ci, ih, iw]) * float(weights[co, ci, r, s])
        measured = float(actual[point])
        finite = math.isfinite(expected) and math.isfinite(measured)
        absolute = abs(measured - expected) if finite else None
        threshold = atol + rtol * abs(expected) if math.isfinite(expected) else None
        rows.append({'position': list(point), 'expected_fp64': expected if math.isfinite(expected) else None,
                     'actual': measured if math.isfinite(measured) else None, 'finite': finite,
                     'absolute_error': absolute, 'relative_error': absolute / max(abs(expected), 1e-12) if finite else None,
                     'threshold': threshold, 'passed': bool(finite and absolute <= threshold)})
    return {'reference': 'independent_python_fp64_points', 'shape_id': shape.id,
            'selection': 'explicit' if points is not None else 'boundaries_groups_seeded_v1',
            'seed': seed, 'atol': atol, 'rtol': rtol, 'point_count': len(rows),
            'output_count': math.prod(shape.output), 'full_output': len(rows) == math.prod(shape.output),
            'passed': all(row['passed'] for row in rows), 'failed_points': sum(not row['passed'] for row in rows),
            'points': rows}


def prepare_reference(shape, x, weights, seed=20260908, count=64):
    """Compute once per input case; reuse immutable positions/FP64 expectations."""
    shape = _shape(shape)
    receipt = evaluate_points(shape, x, weights, np.broadcast_to(0.0, shape.output),
                              seed=seed, count=count)
    return {'reference': receipt['reference'], 'shape_id': shape.id,
            'output_dims': list(shape.output), 'selection': receipt['selection'], 'seed': seed,
            'point_count': receipt['point_count'], 'output_count': receipt['output_count'],
            'full_output': receipt['full_output'],
            'positions': [row['position'] for row in receipt['points']],
            'expected_fp64': [row['expected_fp64'] for row in receipt['points']]}


def check_points(prepared, actual, atol=1e-4, rtol=1e-4):
    """Compare an output to a prepared oracle without recomputing reductions."""
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError('tolerances must be finite and nonnegative')
    actual = _cpu(actual, prepared['output_dims'])
    positions, expected_values = prepared['positions'], prepared['expected_fp64']
    if not positions or len(positions) != len(expected_values):
        raise ValueError('invalid prepared reference')
    rows = []
    for position, expected in zip(positions, expected_values):
        measured = float(actual[tuple(position)])
        finite = expected is not None and math.isfinite(expected) and math.isfinite(measured)
        absolute = abs(measured - expected) if finite else None
        threshold = atol + rtol * abs(expected) if expected is not None else None
        rows.append({'position': list(position), 'expected_fp64': expected,
                     'actual': measured if math.isfinite(measured) else None, 'finite': finite,
                     'absolute_error': absolute, 'relative_error': absolute / max(abs(expected), 1e-12) if finite else None,
                     'threshold': threshold, 'passed': bool(finite and absolute <= threshold)})
    return {key: value for key, value in prepared.items() if key not in ('positions', 'expected_fp64')} | {
        'atol': atol, 'rtol': rtol, 'points': rows, 'passed': all(row['passed'] for row in rows),
        'failed_points': sum(not row['passed'] for row in rows)}
