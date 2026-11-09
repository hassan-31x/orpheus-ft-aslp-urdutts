import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from screening_recovery import copy_cache, restore_inputs, publish, restore_remote
from orpheus_utils import atomic_json, sha256


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.recipe={'data_revision':'fixed','quality':'strict'}
        self.source=self.root/'inputs/prior/orpheus-mixed/datasets/abc'
        self.source.mkdir(parents=True)
        atomic_json(self.source/'selection_progress.json',{'recipe':self.recipe})
        audio=self.source/'screening/urdutts-one.wav';audio.parent.mkdir();audio.write_bytes(b'audio')
        atomic_json(self.source/'tts_scan_state.json',{'done':['shard0'],'pool':[{'screening_path':'screening/urdutts-one.wav','sha256':sha256(audio)}]})
        atomic_json(self.source/'quality_cache/score.json',{'rejection':None})

    def test_previous_output_restores_cursor_audio_and_scores(self):
        target=self.root/'work/datasets/cache';target.parent.mkdir(parents=True)
        self.assertTrue(restore_inputs({},self.recipe,target,self.root/'inputs'))
        self.assertEqual(json.loads((target/'tts_scan_state.json').read_text())['done'],['shard0'])
        self.assertEqual((target/'screening/urdutts-one.wav').read_bytes(),b'audio')
        self.assertTrue((target/'quality_cache/score.json').exists())

    def test_recipe_change_is_not_reused(self):
        target=self.root/'work/cache'
        self.assertFalse(restore_inputs({},dict(self.recipe,quality='different'),target,self.root/'inputs'))
        self.assertFalse(target.exists())

    def test_missing_live_shortlist_audio_rejects_snapshot(self):
        (self.source/'screening/urdutts-one.wav').unlink()
        with self.assertRaisesRegex(RuntimeError,'Shortlist changed'):
            copy_cache(self.source,self.root/'copy',self.recipe)

    def test_completed_selection_does_not_require_discarded_screening_wavs(self):
        (self.source/'screening/urdutts-one.wav').unlink()
        atomic_json(self.source/'dataset_source.json',{'recipe':self.recipe,'selected':{}})
        copy_cache(self.source,self.root/'copy',self.recipe)

    def test_backup_atomic_commit_and_remote_round_trip(self):
        work=self.root/'work';work.mkdir()
        cfg={'run_id':'example','hf_repo_id':'private/repo','work_dir':str(work)}
        blobs={}
        class Api:
            def create_commit(self,**kwargs):
                for op in kwargs['operations']:
                    blobs[op.path_in_repo]=Path(op.path_or_fileobj).read_bytes()
        publish(cfg,self.recipe,self.source,api=Api())
        self.assertEqual(len(blobs),2)
        def download(repo,name,local_dir=None):
            path=Path(local_dir or self.root/'downloads')/name
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blobs[name]);return str(path)
        with patch('huggingface_hub.hf_hub_download',download):
            target=work/'restored';target.parent.mkdir(exist_ok=True)
            self.assertTrue(restore_remote(cfg,self.recipe,target))
            self.assertEqual((target/'screening/urdutts-one.wav').read_bytes(),b'audio')

    def test_symlink_is_rejected(self):
        (self.source/'quality_cache/link').symlink_to(self.source/'selection_progress.json')
        with self.assertRaisesRegex(ValueError,'Symlink'):
            copy_cache(self.source,self.root/'copy',self.recipe)
