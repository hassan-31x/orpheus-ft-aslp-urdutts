"""WAV cleanup must preserve every checkpoint/cache and require complete encoding."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from audio_storage import release_encoded_audio
from orpheus_utils import atomic_json, fingerprint, sha256


class AudioStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.cfg = dict(work_dir=str(self.work), dataset_id="kaab4321/UrduTTS",
                        dataset_revision="a" * 40, chunk_size=2, seed=3407, sample_count=1)
        recipe = dict(dataset_id=self.cfg["dataset_id"], revision=self.cfg["dataset_revision"])
        self.root = self.work / "datasets" / fingerprint(recipe)
        self.run, self.cache = self.work / "runs/r", self.work / "cache/tokens"
        self.run.mkdir(parents=True)
        self.rows = {}
        (self.root / "parts").mkdir(parents=True)
        for split in ("train", "validation"):
            rows = []
            for i in range(6):
                audio = f"audio/{split}/00000-{i:06d}.wav"
                path = self.root / audio
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"generated waveform {split} {i}".encode())
                rows.append(dict(audio=audio, text="اردو", duration=1., sha256=sha256(path)))
            self.rows[split] = rows
            atomic_json(self.root / "parts" / f"{split}.json", dict(
                identity=dict(recipe=self.root.name), rows=rows))
            with (self.root / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["audio", "text"])
                writer.writeheader()
                writer.writerows(dict(audio=r["audio"], text=r["text"]) for r in rows)
            for start in (0, 2, 4):
                path = self.cache / split / f"{start:07d}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                subset = rows[start:start+2]
                pd.DataFrame([dict(audio=r["audio"], input_ids=[1, 2], labels=[1, 2]) for r in subset]).to_parquet(path)
                atomic_json(path.with_suffix(".json"), dict(
                    row_identity=fingerprint(subset), rows=2, excluded=[], sha256=sha256(path)))
        atomic_json(self.root / "dataset_source.json", dict(recipe=recipe, splits={s: {} for s in self.rows}))
        atomic_json(self.run / "normalized_manifests.json", self.rows)
        atomic_json(self.run / "training_manifests.json", self.rows)
        atomic_json(self.run / "token_format.json", dict(splits={s: fingerprint(r) for s, r in self.rows.items()}))
        checkpoint = self.run / "checkpoint-1"
        checkpoint.mkdir()
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer to preserve")

    def tearDown(self):
        self.tmp.cleanup()

    def cleanup(self):
        return release_encoded_audio(self.root, self.cache, self.run, self.cfg)

    def test_only_generated_audio_removed_after_full_verification(self):
        unknown = self.root / "audio/keep-user-file.wav"
        unknown.write_bytes(b"not registered")
        originals = {str(p): sha256(p) for p in self.cache.rglob("*") if p.is_file()}
        report = self.cleanup()
        self.assertGreater(report["removed_files"], 0)
        self.assertGreater(report["removed_bytes"], 0)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(unknown.exists())
        self.assertEqual((self.run / "checkpoint-1/optimizer.pt").read_bytes(), b"optimizer to preserve")
        for path, digest in originals.items():
            self.assertEqual(sha256(path), digest)
        for relative in report["protected_audio"]:
            self.assertTrue((self.root / relative).is_file())
        second = self.cleanup()
        self.assertEqual(second["removed_files"], 0)

    def assert_no_audio_removed(self, message):
        before = sorted(str(p) for p in self.root.rglob("*.wav"))
        with self.assertRaisesRegex(RuntimeError, message):
            self.cleanup()
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*.wav")))

    def test_missing_or_corrupt_token_chunk_blocks_all_deletions(self):
        path = self.cache / "validation/0000004.parquet"
        data = path.read_bytes()
        path.unlink()
        self.assert_no_audio_removed("token chunk")
        path.write_bytes(data + b"corrupt")
        self.assert_no_audio_removed("token chunk")

    def test_wrong_retained_coverage_blocks_all_deletions(self):
        changed = dict(self.rows, train=self.rows["train"][:-1])
        atomic_json(self.run / "training_manifests.json", changed)
        self.assert_no_audio_removed("coverage mismatch")

    def test_changed_wav_is_preserved(self):
        relative = self.rows["train"][-1]["audio"]
        (self.root / relative).write_bytes(b"user changed this")
        report = self.cleanup()
        self.assertIn(relative, report["preserved_changed_audio"])
        self.assertTrue((self.root / relative).is_file())

    def test_external_dataset_or_unsafe_registry_is_rejected(self):
        self.cfg["work_dir"] = str(self.work / "another-work")
        self.assert_no_audio_removed("generated, pinned dataset")
        self.cfg["work_dir"] = str(self.work)
        marker = self.root / "parts/train.json"
        part = json.loads(marker.read_text())
        part["rows"][-1]["audio"] = "../outside.wav"
        atomic_json(marker, part)
        self.assert_no_audio_removed("unexpected generated WAV path")

    def test_registry_must_cover_every_audited_row(self):
        (self.root / "parts/train.json").unlink()
        self.assert_no_audio_removed("incomplete conversion registry")


if __name__ == "__main__":
    unittest.main()
