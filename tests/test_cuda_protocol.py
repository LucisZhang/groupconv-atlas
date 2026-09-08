"""CPU-only tests for CUDA evidence gates and protocol boundaries."""
import copy
import json
import unittest
from bench.api import ROOT
from bench.check import boundary_shapes, expected_support, SEEDS
from bench.cuda_run import validate_correctness, main


class CudaProtocolTests(unittest.TestCase):
    def receipt(self):
        records = []
        for shape in boundary_shapes():
            for impl in range(4):
                for seed in SEEDS:
                    for ref in ('torch_cpu_fp32_full', 'python_fp64_full'):
                        records.append({'backend':'cuda','shape_id':shape.id,'implementation':impl,
                            'seed':seed,'reference':ref,'status':'PASS' if expected_support('cuda',shape,impl) else 'UNSUPPORTED'})
        return {'kind':'correctness','library':{'sha256':'lib'},'source':{'source_tree_sha256':'tree'},
                'availability':{'cuda':{'available':True}}, 'contract_checks':[{'status':'PASS'}],
                'limit':None,'records':records}

    def test_complete_receipt(self):
        validate_correctness(self.receipt(), {'source_tree_sha256':'tree'}, 'lib')

    def test_reject_missing_or_mismatched_evidence(self):
        base = self.receipt()
        mutations = [lambda r:r.pop('source'),lambda r:r['library'].update(sha256='other'),
            lambda r:r['availability']['cuda'].update(available=False),
            lambda r:r['records'][0].update(status='FAIL'), lambda r:r.update(contract_checks=[]),
            lambda r:r.update(records=[]),lambda r:r.update(limit=1),
            lambda r:r['records'][0].update(reference='unknown')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                report = copy.deepcopy(base); mutate(report)
                with self.assertRaises(ValueError): validate_correctness(report, {'source_tree_sha256':'tree'}, 'lib')

    def test_smoke_still_requires_source_and_cuda(self):
        report = self.receipt(); report.pop('source')
        with self.assertRaises(ValueError): validate_correctness(report, {'source_tree_sha256':'tree'}, 'lib', True)

    def test_protocol_separates_graph_and_eager(self):
        protocol = json.loads((ROOT/'configs/cuda-protocol.json').read_text())
        self.assertEqual((protocol['batches'], protocol['samples_per_batch']), (5,30))
        self.assertLessEqual(protocol['repeat_cap'], 100)
        self.assertIn('t_device_op_graph_us', protocol['device_boundary'])
        self.assertEqual(protocol['host_to_host'], 'NOT_RUN')

    def test_short_formal_rejected_before_device_access(self):
        with self.assertRaises(SystemExit) as error:
            main(['--output','/tmp/unused-cuda-test','--correctness','/tmp/missing','--samples','1'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__': unittest.main()
