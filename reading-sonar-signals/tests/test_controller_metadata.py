"""Administrative schema, metadata-boundary and resume checks; no CUDA arithmetic."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os

from sonar.controller import report_schema_errors, restricted_state, compact_state, action_menu

RUNNER=Path(__file__).resolve().parents[1]/'scripts/run_benchmark.py'
spec=importlib.util.spec_from_file_location('sonar_runner_for_tests',RUNNER)
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def report():
    return dict(components=[dict(id='C1',type='tone',frequency_domain='acoustic',
                                time_s=[100.,120.],frequency_hz=[200.,200.],evidence_ids=['E0000'])],
                relations=[],claims=[],unresolved_questions=[],abstained=False)


def state():
    component=dict(id='C0000',type='tone',frequency_domain='acoustic',time_s=[100.,120.],
                   frequency_hz=[200.,200.],evidence_ids=['E0000','E0001'])
    return dict(scene_id='opaque',time_s=[100.,130.],sample_rate_hz=16000.,limitations=[],
                components=[component],relations=[],measurements=[
                    dict(id='E0000',frequency_domain='acoustic',component_ids=['C0000'],
                         observed_components=[dict(component,id='E0000:O000',evidence_ids=['E0000'])],
                         relations=[{'type':'continues','component_ids':['C0000','C0001']}]),
                    dict(id='E0001',frequency_domain='acoustic',component_ids=['C0000'],
                         observed_components=[dict(component,id='E0001:O000',evidence_ids=['E0001'])])])


class ControllerMetadataTests(unittest.TestCase):
    def test_valid_report_and_missing_typed_fields(self):
        self.assertEqual(report_schema_errors(report()),[])
        invalid=report(); invalid.pop('abstained')
        self.assertIn('abstained_must_be_bool',report_schema_errors(invalid))
        invalid=report(); invalid['abstained']=0
        self.assertTrue(report_schema_errors(invalid))
        invalid=report(); invalid['components'][0]['time_s']=[[100.],120.]
        self.assertTrue(report_schema_errors(invalid))
        invalid=report(); invalid['components'][0]['frequency_hz']=[True,200.]
        self.assertTrue(report_schema_errors(invalid))

    def test_chirp_order_and_carrier_requirements(self):
        value=report(); value['components'][0]['type']='chirp'
        self.assertTrue(report_schema_errors(value))
        value['components'][0]['frequency_track']=[[100.,200.],[120.,220.]]
        self.assertEqual(report_schema_errors(value),[])
        value['components'][0]['frequency_track'].reverse()
        self.assertTrue(report_schema_errors(value))
        value=report(); value['components'][0].update(type='modulation',frequency_domain='modulation')
        self.assertTrue(report_schema_errors(value))
        value['components'][0]['carrier_band_hz']=[500.,2000.]
        self.assertEqual(report_schema_errors(value),[])

    def test_duplicate_ids_are_schema_errors(self):
        value=report(); value['components'].append(deepcopy(value['components'][0]))
        self.assertTrue(report_schema_errors(value))

    def test_flat_observations_keep_duplicates_without_canonical_associations(self):
        original=state(); before=deepcopy(original)
        flat=restricted_state(original,'no_graph')
        self.assertEqual([c['id'] for c in flat['components']],['E0000:O000','E0001:O000'])
        visible=compact_state(flat,graph=False)
        self.assertNotIn('relations',visible)
        for entry in visible['measurement_index']:
            self.assertNotIn('component_ids',entry)
            self.assertNotIn('relations',entry)
            self.assertNotIn('observed_components',entry)
        self.assertEqual(original,before)

    def test_missing_raw_observations_fail_instead_of_empty_ablation(self):
        old=state(); old['measurements'][0].pop('observed_components')
        with self.assertRaisesRegex(ValueError,'stale state'):
            restricted_state(old,'no_graph')

    def test_no_demon_retains_acoustic_evidence_and_image_only_hides_numbers(self):
        original=state()
        original['components'].append(dict(original['components'][0],id='C0001',type='modulation',frequency_domain='modulation'))
        original['measurements'].append(dict(id='E0002',frequency_domain='modulation',observed_components=[]))
        visible=restricted_state(original,'no_demon')
        self.assertEqual(len(visible['components']),1)
        self.assertEqual([m['id'] for m in visible['measurements']],['E0000','E0001'])
        image_only=compact_state(visible,quantitative=False,graph=False)
        self.assertNotIn('components',image_only)
        self.assertNotIn('measurement_index',image_only)

    def test_action_bands_respect_nyquist_and_fixed_acoustic_resolution(self):
        menu=action_menu(state(),fixed_resolution=True)
        bands=[a['carrier_band_hz'] for a in menu if a['name']=='inspect_modulation']
        self.assertEqual(bands,[[500.,2000.],[2000.,5000.]])
        self.assertTrue(all(a['window_s']==1. for a in menu if a['name']=='inspect_region'))


class ResumeProvenanceTests(unittest.TestCase):
    def test_same_protocol_resume_keeps_original_start(self):
        with tempfile.TemporaryDirectory() as folder:
            first=dict(started_at_unix=1,manifest_sha256='abc',budget=6,source_sha256={'a':'one'})
            runner.freeze_or_resume(folder,first)
            resumed=runner.freeze_or_resume(folder,dict(first,started_at_unix=2))
            self.assertEqual(resumed['started_at_unix'],1)
            self.assertTrue((Path(folder)/'resume_ledger.jsonl').exists())
            with self.assertRaisesRegex(ValueError,'incompatible'):
                runner.freeze_or_resume(folder,dict(first,budget=4))
            with self.assertRaisesRegex(ValueError,'incompatible'):
                runner.freeze_or_resume(folder,dict(first,source_sha256={'a':'two'}))

    def test_incomplete_episode_preserved_and_wrong_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'episode.json'; path.write_text('{"scene_id":')
            self.assertFalse(runner.episode_is_complete(path,'scene','F'))
            self.assertEqual(len(list(Path(folder).glob('episode.incomplete.*.json'))),1)
            path.write_text(json.dumps(dict(scene_id='different',method='F')))
            with self.assertRaisesRegex(ValueError,'identity'):
                runner.episode_is_complete(path,'scene','F')

    def test_decode_failure_is_recorded_without_launching_model_inference(self):
        class Backend:
            device_audit={'administrative_test_double':True}
            def __init__(self,*args,**kwargs): pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'src/sonar').mkdir(parents=True)
            for name in ['signal','evidence','synthetic','metrics','data','vlm','batching','controller','rendering',
                         'json_protocol','constrained_decoding','prompting','input_interventions']:
                (root/f'src/sonar/{name}.py').write_text('# metadata fixture\n')
            (root/'reports').mkdir()
            (root/'reports/model_revision.json').write_text('{"revision":"fixture"}')
            (root/'manifest.jsonl').write_text(json.dumps({'scene_id':'opaque'})+'\n')
            old=os.getcwd(); os.chdir(root)
            try:
                with patch.object(runner,'LocalVLM',Backend), patch.object(runner,'load_experimental_waveform',side_effect=OSError('fixture decode failure')), \
                     patch.object(runner.torch.cuda,'empty_cache'), patch.object(runner.torch,'set_num_threads'), \
                     patch('sys.argv',['run_benchmark.py','--manifest','manifest.jsonl','--output','output','--methods','G1']):
                    runner.main()
            finally:
                os.chdir(old)
            failure=json.loads((root/'output/opaque/G1/failure.json').read_text())
            self.assertEqual(failure['status'],'failed')
            self.assertIn('fixture decode failure',failure['error'])
            completion=json.loads((root/'output/COMPLETED.json').read_text())
            self.assertEqual(completion['execution_failures'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
