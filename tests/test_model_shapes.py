import copy
import json
import os
import unittest
import subprocess
import sys
from pathlib import Path
from bench.api import Shape
from scripts.export_model_shapes import DEFAULT_PROTOCOL, validate_protocol, export_shapes, operation_record, freeze_record


class ModelShapeTests(unittest.TestCase):
    def isolated(self, method):
        if os.environ.get('GC_MODEL_TEST_CHILD') == '1':
            return False
        script = f"import sys,unittest; sys.path.insert(0,'tests'); suite=unittest.defaultTestLoader.loadTestsFromName('test_model_shapes.ModelShapeTests.{method}'); result=unittest.TextTestRunner().run(suite); sys.exit(not result.wasSuccessful())"
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, 'GC_MODEL_TEST_CHILD': '1'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return True

    def test_frozen_protocol(self):
        protocol = json.loads(DEFAULT_PROTOCOL.read_text())
        validate_protocol(protocol)
        for key, value in [('torchvision_version', '0.29.0'), ('weights', 'DEFAULT'), ('device', 'cuda'), ('batches', [1])]:
            altered = copy.deepcopy(protocol); altered[key] = value
            with self.assertRaises(ValueError): validate_protocol(altered)

    def test_frozen_shapes_are_consumable(self):
        protocol = json.loads(DEFAULT_PROTOCOL.read_text())
        rows = protocol['frozen_shapes']
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['shape_id'] for row in rows}), 12)
        for row in rows:
            shape = Shape.from_record(row)
            self.assertEqual(row['no_bias_geometry_id'], shape.id)
            self.assertEqual(row['shape_id'], row['source_operation_id'] if row['bias'] else shape.id)
            self.assertEqual(row['native_status'], 'UNSUPPORTED' if row['bias'] else 'SUPPORTED')
            self.assertEqual(tuple(row['output_dims']), shape.output)
        self.assertEqual(sum(row['bias'] for row in rows), 4)
        self.assertTrue(any(row['module_path'] == 'features.3.conv.1.0' for row in rows))
        self.assertEqual({row['n'] for row in rows}, {1, 4})
        self.assertTrue(any(row['sh'] == 2 for row in rows))

    def test_bias_semantics_survive_freezing(self):
        shape = Shape(cin=96, cout=96, groups=96, h=56, w=56, r=7, s=7, ph=3, pw=3)
        source = operation_record(shape, True)
        plain = operation_record(shape, False)
        frozen = freeze_record({**source, 'model': 'convnext_tiny', 'module_path': 'features.1.0.block.0'})
        self.assertTrue(frozen['bias'])
        self.assertTrue(frozen['source_operation_attributes']['bias'])
        self.assertEqual(frozen['source_operation_attributes']['bias_dims'], [96])
        self.assertEqual(frozen['native_status'], 'UNSUPPORTED')
        self.assertEqual(frozen['native_reason'], 'bias not implemented')
        self.assertNotEqual(source['source_operation_id'], plain['source_operation_id'])
        self.assertNotEqual(frozen['shape_id'], frozen['no_bias_geometry_id'])
        self.assertEqual(plain['shape_id'], shape.id)

    def test_version_mismatch_rejected(self):
        if self.isolated("test_version_mismatch_rejected"):
            return
        import torchvision
        if str(torchvision.__version__).split('+')[0] == '0.23.0':
            self.skipTest('running the pinned version')
        with self.assertRaisesRegex(RuntimeError, 'version mismatch'):
            export_shapes()

    @unittest.skipUnless(os.environ.get('GC_RUN_MODEL_EXPORT') == '1', 'set GC_RUN_MODEL_EXPORT=1 for real CPU forward')
    def test_real_forward(self):
        if self.isolated("test_real_forward"):
            return
        report = export_shapes()
        self.assertEqual(len(report['shapes']), 12)
        self.assertEqual({r['model'] for r in report['shapes']}, {'mobilenet_v2', 'resnext50_32x4d', 'convnext_tiny'})
        self.assertEqual({r['n'] for r in report['shapes']}, {1, 4})
        self.assertTrue(any(r['sh'] == 2 for r in report['shapes']))
        for row in report['shapes']:
            shape = Shape.from_record(row)
            self.assertEqual(tuple(row['output_dims']), shape.output)
            self.assertEqual(row['no_bias_geometry_id'], shape.id)
            self.assertEqual(row['shape_id'], row['source_operation_id'] if row['bias'] else shape.id)
            self.assertEqual(row['native_status'], 'UNSUPPORTED' if row['bias'] else 'SUPPORTED')
            self.assertEqual(row['shape_source'], 'actual_cpu_forward_hook')
            self.assertIsNone(row['weights'])
        self.assertEqual(len(report['forward_inventory']), 6)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__': unittest.main()
