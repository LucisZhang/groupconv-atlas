"""CPU-only frozen model denominator and worker evidence gates."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from bench.api import ROOT, Shape
from bench.fp64_points import select_points
from bench.model_run import (validate_report_rows, validate_shape_report, excluded_result,
                             validate_worker_result, main, VARIANTS)


class ModelRunTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT/'results/preparation-20260908/model-shapes.json').read_text())
        self.protocol = json.loads((ROOT/'configs/model-shapes-protocol.json').read_text())

    def test_verified_report_and_denominator(self):
        _, rows, digest = validate_shape_report(ROOT/'results/preparation-20260908/model-shapes.json')
        self.assertEqual(len(rows), 12); self.assertEqual(len(digest), 64)
        self.assertEqual(sum(r['bias'] for r in rows), 4)
        self.assertEqual(sum(r['native_status']=='SUPPORTED' for r in rows), 8)
        for row in rows:
            if row['bias']:
                result = excluded_result(row)
                self.assertTrue(result['shape']['bias'])
                self.assertEqual(len(result['samples']), 0)
                self.assertEqual([r['status'] for r in result['rows'][:4]], ['UNSUPPORTED']*4)
                self.assertEqual([r['status'] for r in result['rows'][4:]], ['NOT_RUN']*2)

    def test_bias_shapeid_duplicate_and_missing_rejected(self):
        def bias(r):
            row = next(row for row in r['shapes'] if row['bias']); row['bias'] = False
        changes = [bias, lambda r:r['shapes'][0].update(shape_id='edited'),
                   lambda r:r['shapes'].__setitem__(1, copy.deepcopy(r['shapes'][0])),
                   lambda r:r['shapes'].pop(),
                   lambda r:r['shapes'][0]['source_operation_attributes'].update(bias=True)]
        for change in changes:
            with self.subTest(change=change):
                report = copy.deepcopy(self.report); change(report)
                with self.assertRaises(ValueError): validate_report_rows(report, self.protocol)
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'changed.json'; path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError): validate_shape_report(path)

    def test_formal_counts_rejected_before_device_access(self):
        with self.assertRaises(SystemExit) as error:
            main(['--shape-report','/missing','--correctness','/missing','--output','/tmp/unused-model-run','--samples','1'])
        self.assertEqual(error.exception.code,2)

    def test_worker_pass_requires_complete_evidence(self):
        record = next(r for r in self.report['shapes'] if not r['bias'])
        shape = Shape.from_record(record)
        protocol = {'batches':5,'samples_per_batch':30}
        # A wholly failed worker can be recorded; a PASS cannot omit its proof.
        result = {'rows':[{'shape_id':shape.id,'implementation':name,'status':'FAIL'} for name in VARIANTS], 'samples':[]}
        validate_worker_result(copy.deepcopy(result),record,protocol)
        result['rows'][0]['status']='PASS'
        with self.assertRaisesRegex(ValueError,'FP32'): validate_worker_result(result,record,protocol)
        result['rows'][0]['error']={'passed':True,'elements':shape.n*shape.cout*shape.output[2]*shape.output[3]}
        with self.assertRaisesRegex(ValueError,'FP64'): validate_worker_result(result,record,protocol)
        result['rows'][0]['fp64_points']={'passed':True,'point_count':64,'shape_id':shape.id,'reference':'independent_python_fp64_points',
            'points':[{'passed':True,'position':point} for point in select_points(shape)]}
        with self.assertRaisesRegex(ValueError,'samples'): validate_worker_result(result,record,protocol)
        result['samples']=[{'shape_id':shape.id,'implementation':'k0','batch':b,'sample':s,
            't_device_op_graph_us':1.,'t_api_us':2.,'t_host_feed_us':1.}
            for b in range(5) for s in range(30)]
        self.assertFalse(validate_worker_result(result,record,protocol)['shape']['bias'])
        result['samples'][0]['sample']=1
        with self.assertRaisesRegex(ValueError,'samples'): validate_worker_result(result,record,protocol)

    def test_biased_layer_cannot_reach_worker(self):
        record=next(r for r in self.report['shapes'] if r['bias'])
        with self.assertRaisesRegex(ValueError,'biased'): validate_worker_result({},record,{})


if __name__ == '__main__': unittest.main()
