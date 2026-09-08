import copy
import hashlib
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bench.api import Shape

ORACLE_SHAPE = Shape(n=1,cin=1,cout=1,h=1,w=64,groups=1).record()
from scripts.analyze_supplemental import (analyze, bind_source, compare, fp32, fp64, medians,
                                         native_receipt, strict_precision, unit, contract_receipt,
                                         shape_context, bind_shape_validator)


class SupplementalAnalysis(unittest.TestCase):
    def test_oracle_shape_metadata_must_be_canonical(self):
        shape_context(ORACLE_SHAPE)
        for field, bad in [('dtype','float16'),('layout','NHWC'),('bias',True),
                           ('input_dims',[1,1,1,63]),('weight_dims',[1,1,1,1]),
                           ('output_dims',[True,1,1,64])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                shape_context({**ORACLE_SHAPE,field:bad})

    def test_frontend_shape_validator_is_source_bound(self):
        import bench.api as shape_api
        digest = hashlib.sha256(Path(shape_api.__file__).read_bytes()).hexdigest()
        bind_shape_validator({'source_files':{'bench/api.py':digest}})
        for source in ({},{'source_files':{'bench/api.py':'0'*64}}):
            with self.assertRaisesRegex(ValueError,'shape validator source binding'):
                bind_shape_validator(source)

    def test_batch_median_not_pooled(self):
        rows=[{'batch':b,'sample':s,'t':v} for b,values in enumerate([[1,1,100],[10,10,10]]) for s,v in enumerate(values)]
        self.assertEqual(medians(rows,2,3,['t']),{'t':[1,10]})

    def test_missing_and_duplicate_samples_reject(self):
        rows=[{'batch':0,'sample':0,'t':1}]*2
        with self.assertRaises(ValueError): medians(rows,1,2,['t'])
        with self.assertRaises(ValueError): medians(rows[:1],1,2,['t'])

    def test_nan_and_boolean_reject(self):
        for value in [float('nan'),float('inf'),True,0,-1]:
            with self.assertRaises(ValueError): medians([{'batch':0,'sample':0,'t':value}],1,1,['t'])

    def test_five_batches_cannot_win(self):
        r=compare([100]*5,[1]*5,paired=True)
        self.assertEqual(r['classification'],'UNCERTAIN');self.assertIsNone(r['ci95'])

    def test_unpaired_ten_batches_cannot_win(self):
        r=compare([100]*10,[1]*10,paired=False)
        self.assertEqual(r['classification'],'UNCERTAIN');self.assertIsNone(r['ci95'])

    def test_paired_bootstrap_reuses_same_indices(self):
        r=compare([i*2 for i in range(1,11)],list(range(1,11)),paired=True,draws=1000)
        self.assertEqual(r['ci95'],[2.,2.]);self.assertEqual(r['classification'],'WIN')

    def test_tie_margin(self):
        self.assertEqual(compare([1]*10,[1]*10,paired=True,draws=1000)['classification'],'TIE')

    def test_nonpass_does_not_become_measured(self):
        for status in ['FAIL','UNSUPPORTED','OOM','TIMEOUT','NOT_RUN','RUNNING']:
            value=unit('x',{'status':status},[],{'batches':5,'samples_per_batch':30})
            self.assertEqual(value['assessment'],'NOT_ASSESSED');self.assertNotIn('batch_medians',value)

    def test_empty_pass_rejects(self):
        with self.assertRaises(ValueError):unit('x',{'status':'PASS'},[],{'batches':5,'samples_per_batch':30})

    def test_precision_gate(self):
        strict_precision({'api':'allow_tf32','matmul':False,'cudnn':False})
        for data in [{},{'api':'allow_tf32','matmul':True,'cudnn':False},{'api':'allow_tf32','matmul':0,'cudnn':0}]:
            with self.assertRaises(ValueError):strict_precision(data)

    def test_fp32_gate_denominator(self):
        value={'passed':True,'failed_elements':0,'elements':20,'atol':1e-4,'rtol':1e-4,'max_abs':0.,'max_rel':0.,'rms':0.,'epsilon':1e-12}
        fp32(value,20)
        with self.assertRaises(ValueError):fp32(value,21)
        with self.assertRaises(ValueError):fp32({'passed':True})

    def test_fp64_gate_rejects_forged_pass(self):
        value={'passed':True,'failed_points':0,'point_count':64,'shape_id':ORACLE_SHAPE['shape_id'],'output_count':64,'reference':'independent_python_fp64_points','atol':1e-4,'rtol':1e-4,
               'points':[{'position':[0,0,0,i],'passed':True,'finite':True,'actual':0.,'expected_fp64':0.,'absolute_error':0.,'relative_error':0.,'threshold':1e-4} for i in range(64)]}
        fp64(value,ORACLE_SHAPE)
        bad=copy.deepcopy(value);bad['points'][-1]['actual']=1
        with self.assertRaises(ValueError):fp64(bad,ORACLE_SHAPE)
        bad=copy.deepcopy(value);bad['points'][-1]=bad['points'][0]
        with self.assertRaises(ValueError):fp64(bad,ORACLE_SHAPE)

    def test_fp32_nonfinite_and_inconsistent_summary_reject(self):
        value={'passed':True,'failed_elements':0,'elements':64,'atol':1e-4,'rtol':1e-4,'max_abs':0.,'max_rel':0.,'rms':0.,'epsilon':1e-12}
        for field,bad in [('max_abs',None),('max_rel',float('nan')),('rms',float('inf')),('max_abs',-1),('rms',1)]:
            with self.assertRaises(ValueError):fp32({**value,field:bad},64)

    def test_fp64_shape_and_coordinates_reject(self):
        value={'passed':True,'failed_points':0,'point_count':64,'shape_id':ORACLE_SHAPE['shape_id'],'output_count':64,'reference':'independent_python_fp64_points','atol':1e-4,'rtol':1e-4,
               'points':[{'position':[0,0,0,i],'passed':True,'finite':True,'actual':0.,'expected_fp64':0.,'absolute_error':0.,'relative_error':0.,'threshold':1e-4} for i in range(64)]}
        for position in ([0,0,0,64],[-1,0,0,0],[0,0,0],[False,0,0,0],[0.,0,0,0]):
            bad=copy.deepcopy(value);bad['points'][0]['position']=position
            with self.assertRaises(ValueError):fp64(bad,ORACLE_SHAPE)
        with self.assertRaises(ValueError):fp64({**value,'shape_id':'foreign'},ORACLE_SHAPE)
        with self.assertRaises(ValueError):fp64({**value,'output_count':999},ORACLE_SHAPE)

    def test_single_row_correctness_rejected_by_complete_validator(self):
        import sys
        from bench import cuda_run
        root=Path(__file__).parents[1]
        files={str(Path(m.__file__).resolve().relative_to(root)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for n,m in tuple(sys.modules.items()) if n.startswith('bench.') and getattr(m,'__file__',None)}
        source={'source_tree_sha256':'synthetic-gate-only','source_files':files}
        key={'source':source,'library_sha256':'synthetic-library'}
        report={'kind':'correctness','source':source,'library':{'sha256':'synthetic-library'},'availability':{'cuda':{'available':True}},
                'counts':{'FAIL':0},'contract_counts':{'FAIL':0},'records':[{'backend':'cuda','status':'PASS'}],'contract_checks':[{'status':'PASS'}]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'correctness.json';path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'boundary implementation/seed coverage'):native_receipt(key,path)
            bad=copy.deepcopy(key);bad['source']['source_files']['bench/cuda_run.py']='0'*64
            with self.assertRaisesRegex(ValueError,'validator source binding'):native_receipt(bad,path)

    def test_actual_boundary_patterns_are_distinct_but_duplicates_reject(self):
        # Actual production shape/pattern identities; numerical values below are
        # synthetic. Only the separate complete-coverage gate is mocked here.
        import sys
        from bench import cuda_run
        root=Path(__file__).parents[1]
        files={str(Path(m.__file__).resolve().relative_to(root)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for n,m in tuple(sys.modules.items()) if n.startswith('bench.') and getattr(m,'__file__',None)}
        source={'source_tree_sha256':'synthetic-identity-check','source_files':files}
        key={'source':source,'library_sha256':'synthetic-library'}
        metrics={'passed':True,'failed_elements':0,'elements':64,'atol':1e-4,'rtol':1e-4,'max_abs':0.,'max_rel':0.,'rms':0.,'epsilon':1e-12}
        records=[{'suite':'boundary','backend':'cuda','shape_id':'gc_2c9ee62d82a63099','implementation':0,'threads':None,'seed':None,'input_pattern':pattern,'reference':'torch_cpu_fp32_full','status':'PASS','metrics':metrics}
                 for pattern in ('zeros','impulse','group_isolation','cancellation')]
        expected=dict(dtype=1,rank=1,strides=2,misaligned=1,device=1,buffer_bytes=1,alias=1,groups=1,geometry=1,implementation=1)
        checks=[{'case':k,'status':'PASS','actual_status':v,'expected_status':v} for k,v in expected.items()]
        checks += [{'case':k,'status':'PASS'} for k in ['reject_nonfinite_nan','reject_nonfinite_inf','reject_nonfinite_-inf','python_shape_overflow','python_impl_overflow','python_nonintegral']]
        report={'kind':'correctness','source':source,'library':{'sha256':'synthetic-library'},'availability':{'cuda':{'available':True}},'limit':None,'counts':{'PASS':4,'FAIL':0},'records':records,'contract_checks':checks}
        with tempfile.TemporaryDirectory() as folder, patch.object(cuda_run,'validate_correctness'):
            path=Path(folder)/'correctness.json';path.write_text(json.dumps(report))
            native_receipt(key,path)
            report['records'].append(copy.deepcopy(records[0]));report['counts']['PASS']=5
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'duplicate correctness record'):native_receipt(key,path)

    def test_contract_denominator_and_codes_reject(self):
        expected=dict(dtype=1,rank=1,strides=2,misaligned=1,device=1,buffer_bytes=1,alias=1,groups=1,geometry=1,implementation=1)
        rows=[{'case':k,'status':'PASS','actual_status':v,'expected_status':v} for k,v in expected.items()]
        rows += [{'case':k,'status':'PASS'} for k in ['reject_nonfinite_nan','reject_nonfinite_inf','reject_nonfinite_-inf','python_shape_overflow','python_impl_overflow','python_nonintegral']]
        contract_receipt(rows)
        with self.assertRaises(ValueError):contract_receipt(rows[:1])
        bad=copy.deepcopy(rows);bad[0]['actual_status']=0
        with self.assertRaises(ValueError):contract_receipt(bad)

    def test_source_snapshot_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'a.py').write_text('x=1\n')
            files={'a.py':hashlib.sha256((root/'a.py').read_bytes()).hexdigest()}
            tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
            source={'source_identity_version':2,'source_files':files,'source_tree_sha256':tree}
            bind_source(source,{'source_tree_sha256':tree},root)
            (root/'a.py').write_text('x=2\n')
            with self.assertRaises(ValueError):bind_source(source,{'source_tree_sha256':tree},root)

    def test_correctness_cannot_be_missing(self):
        with self.assertRaises(ValueError):native_receipt({},None)

    def test_frontend_unsupported_preserves_four_path_units(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);snapshot=root/'snapshot';snapshot.mkdir();(snapshot/'a.py').write_text('x=1')
            files={'a.py':hashlib.sha256((snapshot/'a.py').read_bytes()).hexdigest()}
            tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
            plan={'schema_version':1,'kind':'frontend','source_tree_sha256':tree,'batches':5,'samples_per_batch':30,'shape_ids':['s'],'layouts':['nchw','channels_last'],
                  'expected_units':[f's/{l}/{p}' for l in ['nchw','channels_last'] for p in ['resident','caller_nchw']]}
            (snapshot/'configs').mkdir();(snapshot/'configs/atlas.json').write_text(json.dumps({'shapes':[{'shape_id':'s'}]}))
            key={'shape_set':'atlas','shape_config_sha256':hashlib.sha256((snapshot/'configs/atlas.json').read_bytes()).hexdigest(),'source':{'source_identity_version':2,'source_files':files,'source_tree_sha256':tree},'protocol':{'batches':5,'samples_per_batch':30},'measurement_kind':'FORMAL','shape_ids':['s'],'layouts':plan['layouts']}
            (root/'run-key.json').write_text(json.dumps(key));(root/'bench.json').write_text(json.dumps({'run_key':key,'records':[{'shape_id':'s','layout':l,'status':'UNSUPPORTED'} for l in plan['layouts']]}))
            result=analyze('frontend',root,plan,snapshot)
            self.assertEqual(result['planned_units'],4);self.assertEqual(result['execution_counts'],{'UNSUPPORTED':4})
            self.assertTrue(all(v['assessment']=='NOT_ASSESSED' for v in result['comparisons']))
            # A missing layout remains explicit NOT_RUN, not a smaller denominator.
            (root/'bench.json').write_text(json.dumps({'run_key':key,'records':[{'shape_id':'s','layout':'nchw','status':'UNSUPPORTED'}]}))
            self.assertEqual(analyze('frontend',root,plan,snapshot)['execution_counts'],{'UNSUPPORTED':2,'NOT_RUN':2})

    def test_frontend_config_order_preserves_exact_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);snapshot=root/'snapshot';(snapshot/'configs').mkdir(parents=True)
            config=snapshot/'configs/atlas.json';config.write_text(json.dumps({'shapes':[{'shape_id':'b'},{'shape_id':'a'}]}))
            files={'configs/atlas.json':hashlib.sha256(config.read_bytes()).hexdigest()}
            tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
            plan={'schema_version':1,'kind':'frontend','source_tree_sha256':tree,'batches':5,'samples_per_batch':30,
                  'shape_ids':['a','b'],'layouts':['nchw'], 'expected_units':[f'{s}/nchw/{p}' for s in ['a','b'] for p in ['resident','caller_nchw']]}
            key={'shape_set':'atlas','shape_config_sha256':files['configs/atlas.json'],
                 'source':{'source_identity_version':2,'source_files':files,'source_tree_sha256':tree},
                 'protocol':{'batches':5,'samples_per_batch':30},'measurement_kind':'FORMAL','shape_ids':['b','a'],'layouts':['nchw']}
            def save():
                (root/'run-key.json').write_text(json.dumps(key))
                (root/'bench.json').write_text(json.dumps({'run_key':key,'records':[{'shape_id':s,'layout':'nchw','status':'UNSUPPORTED'} for s in ['a','b']]}))
            save();result=analyze('frontend',root,plan,snapshot)
            self.assertEqual(result['planned_units'],4);self.assertEqual(result['execution_counts'],{'UNSUPPORTED':4})
            for bad in (['a','a'],['a'],['a','foreign']):
                key['shape_ids']=bad;save()
                with self.subTest(ids=bad),self.assertRaisesRegex(ValueError,'denominator'):analyze('frontend',root,plan,snapshot)

    def test_module_complete_paired_run_and_order_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);snapshot=root/'snapshot';(snapshot/'bench').mkdir(parents=True)
            script=snapshot/'bench/module_run.py';script.write_text('# fixture only')
            files={'bench/module_run.py':hashlib.sha256(script.read_bytes()).hexdigest()}
            tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
            source={'source_identity_version':2,'source_files':files,'source_tree_sha256':tree}
            (root/'random-state.pt').write_bytes(b'synthetic fixture, not a torch model')
            full={'passed':True,'failed_elements':0,'elements':64,'atol':1e-4,'rtol':1e-4,'max_abs':0.,'max_rel':0.,'rms':0.,'epsilon':1e-12}
            points={'passed':True,'failed_points':0,'point_count':64,'shape_id':ORACLE_SHAPE['shape_id'],'output_count':64,'reference':'independent_python_fp64_points','atol':1e-4,'rtol':1e-4,
                    'points':[{'position':[0,0,0,i],'passed':True,'finite':True,'actual':0.,'expected_fp64':0.,'absolute_error':0.,'relative_error':0.,'threshold':1e-4} for i in range(64)]}
            names=['torch_block_tuned','block_one_k3_substitution']
            key={'source':source,'measurement_kind':'DESCRIPTIVE','batches':10,'samples_per_batch':30,'library_sha256':'fixture-library',
                 'model':'torchvision.models.mobilenet_v2','block_path':'features.3','fixed_implementation':'k3','batch_size':1,
                 'precision':{'api':'allow_tf32','matmul':False,'cudnn':False},'gpu':{'name':'SYNTHETIC'},'native_stream':{'non_default':True},
                 'input_sha256':'fixture-input','state_file_sha256':hashlib.sha256((root/'random-state.pt').read_bytes()).hexdigest(),
                 'script_sha256':files['bench/module_run.py'],'random_seed':17,
                 'selected_shape':ORACLE_SHAPE,'observed':{'conv':{'output':{'shape':[1,1,1,64]}},'block':{'output':{'shape':[1,1,1,64]}}},
                 'checks':dict(leaf_conv=full,complete_block=full,complete_model={**full,'elements':1000},leaf_fp64_points=points),
                 'measurements':{n:{'status':'PASS','graph_vs_eager':full} for n in names}}
            plan={'schema_version':1,'kind':'module','source_tree_sha256':tree,'batches':10,'samples_per_batch':30,'batch_size':1,'expected_units':names}
            receipt={'kind':'correctness','source':source,'library':{'sha256':'fixture-library'},'availability':{'cuda':{'available':True}},'counts':{'FAIL':0},
                     'records':[{'backend':'cuda','status':'PASS'}],'contract_checks':[{'status':'PASS'}]}
            cor=root/'correctness.json';cor.write_text(json.dumps(receipt));(root/'summary.json').write_text(json.dumps(key))
            raw=[]
            for b in range(10):
                order=list(names);random.Random(f'17:{b}').shuffle(order)
                for n in names:
                    for i in range(30):raw.append({'implementation':n,'batch':b,'sample':i,'order':order.index(n),
                        't_device_op_graph_us':2 if n==names[0] else 1,'t_api_us':4 if n==names[0] else 2,'t_host_feed_us':1})
            path=root/'samples.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in raw))
            with patch('scripts.analyze_supplemental.native_receipt'):
                result=analyze('module',root,plan,snapshot,cor,draws=1000)
            self.assertEqual(result['execution_counts'],{'PASS':2})
            self.assertTrue(all(c['classification']=='WIN' and c['paired'] for c in result['comparisons']))
            raw[0]['order']=1-raw[0]['order'];path.write_text(''.join(json.dumps(r)+'\n' for r in raw))
            with patch('scripts.analyze_supplemental.native_receipt'), self.assertRaisesRegex(ValueError,'order mismatch'):
                analyze('module',root,plan,snapshot,cor,draws=1000)

    def test_default_plan_has_complete_denominators(self):
        plan=json.loads((Path(__file__).parents[1]/'docs/execution/supplemental-analysis-plan-031887c.json').read_text())
        self.assertEqual({k:len(v['expected_units']) for k,v in plan['runs'].items()}, {'boundaries':288,'models':72,'module-n1':2,'module-n4':2,'frontend':32})
        self.assertEqual(len(plan['runs']['models']['biased_original_layers']),4)
        from bench.model_run import REPORT_HASH, PROTOCOL_HASH
        self.assertEqual(plan['runs']['models']['shape_report_sha256'], REPORT_HASH)
        self.assertEqual(plan['runs']['models']['shape_protocol_sha256'], PROTOCOL_HASH)


if __name__=='__main__':unittest.main()
