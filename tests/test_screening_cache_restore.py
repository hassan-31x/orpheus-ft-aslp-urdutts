"""Reject incompatible screening caches while restoring the transport-only retry."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from mixed_data import recipe_for
from orpheus_utils import atomic_json, fingerprint
from restore_screening_cache import OLD_TRANSPORT, NEW_TRANSPORT, restore


class ScreeningCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = json.loads(Path('configs/mixed.json').read_text())
        self.expected = recipe_for(self.cfg)
        self.previous = copy.deepcopy(self.expected)
        self.previous['source_sha256']['urdutts_data.py'] = OLD_TRANSPORT
        self.source = self.root/'inputs/prior/orpheus-mixed/datasets/old'
        self.source.mkdir(parents=True)
        (self.source/'quality_cache').mkdir()
        (self.source/'quality_cache'/('a'*64+'.json')).write_text('{}')
        (self.source/'screening').mkdir()
        (self.source/'screening'/('aslp-'+'b'*24+'.wav')).write_bytes(b'reference')
        (self.source/'screening'/('urdutts-'+'b'*24+'.wav')).write_bytes(b'exclude')

    def attempt(self):
        atomic_json(self.source/'selection_progress.json', {'recipe': self.previous})
        return restore(self.cfg, self.root/'inputs', self.root/'work',
                       lambda cfg: self.expected, fingerprint, atomic_json)

    def test_transport_only_cache_restore_filters_files_and_records_provenance(self):
        result = self.attempt()
        self.assertEqual(result['status'], 'compatible_cache_restored')
        self.assertEqual(result['quality_records'], 1)
        self.assertEqual(result['aslp_files'], 1)
        destination = Path(result['destination'])
        self.assertFalse(list((destination/'screening').glob('urdutts-*')))
        record = json.loads((destination/'screening_cache_restore.json').read_text())
        self.assertEqual(record['previous_recipe'], self.previous)
        self.assertEqual(record['recipe'], self.expected)

    def test_quality_data_code_packages_and_smoke_changes_prevent_reuse(self):
        changes = [lambda r: r['mixture'].update(maximum_cer=.5),
                   lambda r: r.update(revision='a'*40),
                   lambda r: r['source_sha256'].update(**{'quality.py':'b'*64}),
                   lambda r: r['packages'].update(numpy='different'),
                   lambda r: r.update(seed=42),
                   lambda r: r.update(test_train_clips=20)]
        baseline = copy.deepcopy(self.previous)
        for change in changes:
            with self.subTest(change=changes.index(change)):
                self.previous = copy.deepcopy(baseline)
                change(self.previous)
                self.assertEqual(self.attempt()['status'], 'no_matching_preparation_cache')

    def test_existing_selected_dataset_is_left_untouched(self):
        destination = self.root/'work/datasets'/fingerprint(self.expected)
        atomic_json(destination/'dataset_source.json', {'selected':'fixed'})
        self.assertEqual(self.attempt()['status'], 'selected_dataset_already_exists')
        self.assertEqual(json.loads((destination/'dataset_source.json').read_text()), {'selected':'fixed'})

    def test_other_transport_versions_cannot_use_this_migration(self):
        self.expected['source_sha256']['urdutts_data.py'] = 'f'*64
        with self.assertRaisesRegex(RuntimeError, 'verified transport fix'):
            self.attempt()
