"""Exercise real Arrow embedded audio, resampling, source selection and recovery."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import soundfile as sf

from finetune_mixed import DEFAULTS, audit_data
from urdutts_data import convert_audio, prepare_dataset, stream_rows, validate_config


def example(index=0, rate=44100, channels=1):
    data = io.BytesIO()
    samples = (0.2 * np.sin(np.arange(rate // 10) * (index + 1) * 0.03)).astype('float32')
    if channels == 2:
        samples = np.stack([samples, samples * 0.5], axis=1)
    sf.write(data, samples, rate, format='WAV', subtype='PCM_16')
    return dict(audio=dict(bytes=data.getvalue(), path='/creator/machine/ignored.wav'),
                transcription=f'یہ اردو مثال ہے {index}', roman='DO NOT USE', phonemes='DO NOT USE')


class UrduTTSDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.cfg = json.loads(DEFAULTS.read_text())
        self.cfg.update(test_train_clips=2, test_validation_clips=1, run_id='urdutts-smoke')
        self.requests = []

    def tearDown(self):
        self.tmp.cleanup()

    def metadata(self, filename):
        self.requests.append(filename)
        if filename == 'dataset_dict.json':
            return {'splits': ['train', 'eval', 'test']}
        return {'_data_files': [{'filename': 'data-00000.arrow'}, {'filename': 'data-00001.arrow'}]}

    def reader(self, filename, limit):
        self.requests.append(filename)
        offset = 10 if filename.startswith('eval/') else 20 if filename.startswith('test/') else 0
        for index in range(min(limit or 3, 3)):
            yield index, example(index + offset)

    def prepare(self, **overrides):
        return prepare_dataset(dict(self.cfg, **overrides), self.work,
                               metadata=self.metadata, row_reader=self.reader)

    def test_native_rate_resampled_without_trimming_and_stereo_averaged(self):
        for rate in (22050, 24000, 44100, 48000):
            destination = self.work / f'{rate}.wav'
            props = convert_audio(example(rate=rate, channels=2)['audio'], destination)
            audio, actual = sf.read(destination)
            self.assertEqual(actual, 24000)
            self.assertEqual(len(audio), 2400)
            self.assertEqual(sf.info(destination).subtype, 'PCM_16')
            self.assertEqual(props['source_channels'], 2)
            self.assertAlmostEqual(len(audio) / actual, props['duration_seconds'], places=4)

    def test_released_eval_maps_to_validation_and_test_not_requested(self):
        root = self.prepare()
        report = json.loads((root / 'dataset_source.json').read_text())
        self.assertEqual(report['splits']['train']['clips'], 2)
        self.assertEqual(report['splits']['validation']['clips'], 1)
        self.assertFalse(any(p.startswith('test/') for p in self.requests))
        self.assertFalse((root / 'test.csv').exists())
        with (root / 'train.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]['text'], example()['transcription'])
        run = self.work / 'run'; run.mkdir()
        splits = audit_data(root, self.cfg, run)
        self.assertEqual({s: len(r) for s, r in splits.items()}, {'train': 2, 'validation': 1})

    def test_shards_reused_and_corruption_repaired(self):
        root = self.prepare()
        self.requests.clear()
        self.assertEqual(self.prepare(), root)
        self.assertFalse(any(p.endswith('.arrow') for p in self.requests))
        (root / 'audio/train/00000-000000.wav').write_bytes(b'corrupt')
        self.requests.clear()
        self.prepare()
        self.assertIn('train/data-00000.arrow', self.requests)
        self.assertNotIn('eval/data-00000.arrow', self.requests)

    def test_smoke_only_visits_needed_shards_and_full_has_separate_cache(self):
        smoke = self.prepare()
        self.assertFalse(any('00001.arrow' in p for p in self.requests))
        full = self.prepare(test_train_clips=None, test_validation_clips=None)
        self.assertNotEqual(smoke, full)
        self.assertIn('train/data-00001.arrow', self.requests)

    def test_bad_audio_recorded_but_disk_error_propagates(self):
        def bad(filename, limit):
            yield 0, example(10 if filename.startswith('eval') else 0)
            yield 1, dict(audio={'bytes': b'invalid'}, transcription='اردو')
        root = prepare_dataset(self.cfg, self.work, metadata=self.metadata, row_reader=bad)
        report = json.loads((root / 'dataset_source.json').read_text())
        self.assertEqual(len(report['conversion_exclusions']), 2)
        with patch('urdutts_data.convert_audio', side_effect=OSError(28, 'disk full')):
            with self.assertRaises(OSError):
                self.prepare(test_train_clips=3)

    def test_missing_embedded_audio_does_not_open_creator_path(self):
        with self.assertRaisesRegex(ValueError, 'embedded audio'):
            convert_audio({'path': '/etc/passwd'}, self.work / 'out.wav')

    def test_invalid_configuration_rejected(self):
        for overrides in ({'dataset_revision': 'main'}, {'dataset_text_column': 'roman'},
                          {'data_dir': '/other'}, {'speaker_column': 'speaker'},
                          {'dataset_include_test': 'false'}):
            with self.assertRaises(ValueError):
                validate_config(dict(self.cfg, **overrides))

    def test_real_ipc_stream_reads_embedded_bytes_and_limits_rows(self):
        table = pa.Table.from_pylist([example(), example(1)])
        buffer = io.BytesIO()
        with pa.ipc.new_stream(buffer, table.schema) as writer:
            writer.write_table(table)
        response = unittest.mock.MagicMock()
        response.raw = io.BytesIO(buffer.getvalue())
        response.__enter__.return_value = response
        with patch('requests.get', return_value=response) as get:
            rows = list(stream_rows(self.cfg, 'train/a.arrow', limit=1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1]['transcription'], example()['transcription'])
        self.assertIn(self.cfg['dataset_revision'], get.call_args.args[0])
        self.assertTrue(get.call_args.kwargs['stream'])


if __name__ == '__main__':
    unittest.main()
