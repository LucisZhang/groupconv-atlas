import dataclasses
import json
import unittest
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from bench.api import Shape
from bench.fp64_points import evaluate_points, select_points, prepare_reference, check_points


class FP64PointsTests(unittest.TestCase):
    def test_deterministic_bounded_unique_and_full(self):
        shape = Shape(n=4, cin=4, cout=6, h=7, w=11, groups=2)
        a = select_points(shape, 64, 9)
        self.assertEqual(a, select_points(shape, 64, 9))
        self.assertEqual(len(a), len(set(a)))
        self.assertNotEqual(a, select_points(shape, 64, 10))
        tiny = Shape(cin=1, cout=1, h=1, w=1)
        self.assertEqual(select_points(tiny), [(0, 0, 0, 0)])

    def test_independent_against_cpu_torch_general_geometry(self):
        if os.environ.get('GC_FP64_TORCH_CHILD') != '1':
            script = "import sys,unittest; sys.path.insert(0,'tests'); suite=unittest.defaultTestLoader.loadTestsFromName('test_fp64_points.FP64PointsTests.test_independent_against_cpu_torch_general_geometry'); result=unittest.TextTestRunner().run(suite); sys.exit(not result.wasSuccessful())"
            result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, 'GC_FP64_TORCH_CHILD': '1'}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        import torch
        import torch.nn.functional as F
        rng = np.random.default_rng(51)
        for kernel in (3, 5, 7):
            for batch in (1, 4):
                shape = Shape(n=batch, cin=4, cout=6, h=15, w=17, groups=2,
                              r=kernel, s=kernel, sh=2, sw=1, dh=1, dw=2, ph=1, pw=2)
                x = torch.from_numpy(rng.normal(size=shape.input))
                w = torch.from_numpy(rng.normal(size=shape.weight))
                y = F.conv2d(x, w, stride=(2, 1), dilation=(1, 2), padding=(1, 2), groups=2)
                receipt = evaluate_points(shape, x, w, y, atol=1e-10, rtol=1e-10)
                self.assertTrue(receipt['passed'])
                json.dumps(receipt, allow_nan=False)

    def test_explicit_oracle_and_corruption_and_nonfinite(self):
        shape = Shape(n=4, cin=2, cout=4, h=2, w=3, groups=2, r=1, s=1, ph=0, pw=0)
        x = np.arange(np.prod(shape.input)).reshape(shape.input)
        w = np.ones(shape.weight)
        y = np.repeat(x, 2, axis=1).astype(float)
        good = evaluate_points(shape, x.tolist(), w, y, count=1000)
        self.assertTrue(good['passed']); self.assertTrue(good['full_output'])
        y[0, 0, 0, 0] = np.nan
        bad = evaluate_points(shape, x, w, y, points=[(0, 0, 0, 0)])
        self.assertFalse(bad['passed']); json.dumps(bad, allow_nan=False)
        y[0, 0, 0, 0] = 5
        self.assertFalse(evaluate_points(shape, x, w, y, points=[(0, 0, 0, 0)])['passed'])

    def test_prepared_reference_reuse(self):
        shape = Shape(n=4, cin=2, cout=4, h=2, w=3, groups=2, r=1, s=1, ph=0, pw=0)
        x = np.arange(np.prod(shape.input)).reshape(shape.input)
        w = np.ones(shape.weight)
        y = np.repeat(x, 2, axis=1).astype(float)
        prepared = prepare_reference(shape, x, w, seed=9, count=64)
        frozen = json.dumps(prepared, sort_keys=True, allow_nan=False)
        self.assertTrue(check_points(prepared, y)['passed'])
        self.assertFalse(check_points(prepared, y + 1)['passed'])
        self.assertEqual(frozen, json.dumps(prepared, sort_keys=True, allow_nan=False))

    def test_cuda_inputs_require_explicit_transfer(self):
        from types import SimpleNamespace
        class CudaLike:
            device = SimpleNamespace(type='cuda')
            def detach(self):
                raise AssertionError('must reject before attempting transfer')
        shape = Shape(cin=1, cout=1, h=1, w=1)
        with self.assertRaisesRegex(ValueError, 'copy tensor to CPU'):
            prepare_reference(shape, CudaLike(), np.ones(shape.weight))

    def test_reject_invalid_inputs(self):
        shape = Shape(cin=1, cout=1, h=1, w=1)
        for count in (0, -1):
            with self.assertRaises(ValueError): select_points(shape, count)
        with self.assertRaises(ValueError): select_points(dataclasses.replace(shape, groups=2))
        x, w, y = np.ones(shape.input), np.ones(shape.weight), np.ones(shape.output)
        for points in ([], [(0, 0, 1, 0)], [(0, 0, 0, 0)] * 2):
            with self.assertRaises(ValueError): evaluate_points(shape, x, w, y, points=points)
        with self.assertRaises(ValueError): evaluate_points(shape, x, w, y, atol=-1)


if __name__ == '__main__': unittest.main()
