"""CPU-only contract tests; these do not certify cuDNN execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bench.cudnn_frontend_run import (PROTOCOL, Unsupported, best_candidate, build_failure,
    dependencies, oracle_gate, report, strides, validate_protocol)


class FrontendProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(PROTOCOL.read_text())

    def test_formal_and_smoke_bounds(self):
        validate_protocol(self.p)
        self.p['batches'] = 1
        with self.assertRaises(ValueError): validate_protocol(self.p)
        validate_protocol(self.p, smoke=True)
        for field, value in [('candidate_cap',0), ('shape_timeout_seconds',float('inf')),
                             ('workspace_bytes',-1), ('samples_per_batch',1.5)]:
            broken = copy.deepcopy(self.p); broken[field] = value
            with self.assertRaises(ValueError): validate_protocol(broken, smoke=True)

    def test_precision_and_version_are_not_silently_relaxed(self):
        for field, value in [('frontend_version','2.0'), ('dtype','float16'),
                             ('excluded_numerical_notes',['TENSOR_CORE']),('fp64_points',32),
                             ('cudnn_version',91100),('heuristic_modes',['B']),('source_identity_version',1)]:
            broken = copy.deepcopy(self.p); broken[field] = value
            with self.assertRaises(ValueError): validate_protocol(broken)

    def test_explicit_layout_handles_singleton_channel(self):
        self.assertEqual(strides((4,16,7,5),'nchw'), (560,35,5,1))
        self.assertEqual(strides((4,16,7,5),'channels_last'), (560,1,80,16))
        self.assertEqual(strides((16,1,3,3),'channels_last'), (9,1,3,1))
        with self.assertRaises(ValueError): strides((1,2,3,4),'NHWC')

    def test_missing_vendor_is_unsupported(self):
        with patch('bench.cudnn_frontend_run.importlib.import_module', side_effect=ImportError('missing vendor')):
            with self.assertRaisesRegex(Unsupported, 'unavailable'): dependencies(self.p)

    def test_missing_numeric_note_fails_closed(self):
        torch = SimpleNamespace(__version__='2.8.0+cu128',version=SimpleNamespace(cuda='12.8'),
            cuda=SimpleNamespace(is_available=lambda:True),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda:91002)))
        cudnn = SimpleNamespace(__version__='1.18.0',pygraph=object,create_handle=lambda:1,
            set_stream=lambda *a:None,get_stream=lambda h:1,backend_version=lambda:91002,
            numerical_note=SimpleNamespace(TENSOR_CORE=1),heur_mode=SimpleNamespace(A=1,FALLBACK=3))
        with patch('bench.cudnn_frontend_run.importlib.import_module', side_effect=[torch,cudnn]):
            with self.assertRaisesRegex(Unsupported,'strict FP32 exclusion unavailable'): dependencies(self.p)

    def test_fp64_gate_detects_corruption_shared_by_framework_and_candidate(self):
        import numpy as np
        from bench.api import Shape, inputs
        from bench.fp64_points import prepare_reference
        s = Shape(n=1,cin=1,cout=1,h=3,w=3)
        x,w = inputs(s)
        prepared = prepare_reference(s,x,w,count=64)
        actual = np.zeros(s.output,dtype=np.float32)
        for position,value in zip(prepared['positions'],prepared['expected_fp64']): actual[tuple(position)] = value
        self.assertTrue(oracle_gate(actual,actual,prepared,self.p)['passed'])
        bad = actual.copy(); bad.flat[0] += 1
        gate = oracle_gate(bad,bad,prepared,self.p)
        self.assertTrue(gate['error']['passed'])
        self.assertFalse(gate['fp64_points']['passed'])
        self.assertFalse(gate['passed'])

    def test_select_only_valid_successful_candidates(self):
        candidates = [{'index':0,'status':'FAIL','tuning_median_graph_us':0.1},
                      {'index':1,'status':'PASS','tuning_median_graph_us':float('nan')},
                      {'index':2,'status':'PASS','tuning_median_graph_us':3.0},
                      {'index':3,'status':'NOT_RUN'},
                      {'index':4,'status':'PASS','tuning_median_graph_us':2.0}]
        self.assertEqual(best_candidate(candidates)['index'],4)
        self.assertIsNone(best_candidate(candidates[:2]))

    def test_vendor_filter_rejection_is_distinct_from_execution_failure(self):
        self.assertEqual(build_failure(RuntimeError('Deselecting execution plan')), 'UNSUPPORTED')
        self.assertEqual(build_failure(RuntimeError('Skipping plan since workspace violation. Requires 999')), 'UNSUPPORTED')
        self.assertEqual(build_failure(RuntimeError('illegal memory access')), 'FAIL')

    def test_report_retains_failure_and_separate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{'shape_id':'s','layout':'nchw','status':'UNSUPPORTED','reason':'no plan'},
                    {'shape_id':'s','layout':'channels_last','status':'FAIL','reason':'capture'}]
            report(root, {'measurement_kind':'SMOKE'}, rows)
            self.assertEqual(json.loads((root/'bench.json').read_text())['records'],rows)
            self.assertIn('t_device_op_graph_us,t_api_us,t_host_feed_us', (root/'bench.csv').read_text())
            self.assertEqual(len((root/'bench.csv').read_text().splitlines()),3)


if __name__ == '__main__': unittest.main()
