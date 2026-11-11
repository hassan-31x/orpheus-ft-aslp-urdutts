import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage2_data
import stage2_plan


def flac(seconds=1.0, rate=24000, seed=0):
    import numpy as np
    import soundfile as sf
    out = io.BytesIO()
    sf.write(out, (0.1 * np.sin(np.arange(int(seconds * rate)) * (0.01 + 0.0007 * seed))).astype("float32"), rate, format="FLAC", subtype="PCM_16")
    return out.getvalue()


def make_shard(path, split, start, count):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [dict(audio=dict(bytes=flac(1.0 + i % 2, seed=start + i + (50 if split == "validation" else 0)), path="x.flac"), id=f"{split}-{start + i}",
                 source="aslp" if i % 2 else "urdutts", category="drama" if i % 2 else "audiobook",
                 text=f"متن {start + i}", duration=1.0 + i % 2, error=None) for i in range(count)]
    pq.write_table(pa.Table.from_pylist(rows), path)


class Stage2DataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.shards = Path(self.tmp.name) / "hub"
        for sub in ("train", "validation"):
            (self.shards / sub).mkdir(parents=True)
        make_shard(self.shards / "train/train-00000-of-00002.parquet", "train", 0, 6)
        make_shard(self.shards / "train/train-00001-of-00002.parquet", "train", 6, 6)
        make_shard(self.shards / "validation/validation-00000-of-00001.parquet", "validation", 0, 4)
        (self.shards / "labels").mkdir()
        with open(self.shards / "labels/v3.jsonl", "w", encoding="utf-8") as f:
            for split, n in (("train", 12), ("validation", 4)):
                for i in range(n):  # drop every 4th train clip; tag the rest
                    f.write(json.dumps(dict(id=f"{split}-{i}", split=split, keep=not (split == "train" and i % 4 == 3),
                                            voice=f"spk_{i} | female | conversational | happy")) + "\n")
        base = json.loads((Path(__file__).resolve().parents[1] / "configs/stage2.json").read_text())
        self.labelled = base
        self.cfg = stage2_plan.epoch_config(2, base)  # plain-text prompts, v2 data
        self.cfg["work_dir"] = str(Path(self.tmp.name) / "work")
        os.environ["HF_TOKEN"] = "test"

    def fake_hub(self):
        shards = self.shards

        class Api:
            def __init__(self, token=None):
                pass

            def list_repo_files(self, *a, **k):
                return [str(p.relative_to(shards)) for p in sorted(shards.rglob("*")) if p.is_file()]

        def download(repo, name, **k):
            return str(shards / name)
        return mock.patch("huggingface_hub.HfApi", Api), mock.patch("huggingface_hub.hf_hub_download", download)

    def prepare(self, cfg):
        a, b = self.fake_hub()
        with a, b:
            return stage2_data.prepare_dataset(cfg, cfg["work_dir"])

    def test_full_preparation_layout_and_resume(self):
        root = self.prepare(self.cfg)
        report = json.loads((root / "dataset_source.json").read_text())
        self.assertEqual(report["splits"]["train"]["clips"], 12)
        self.assertEqual(report["splits"]["validation"]["clips"], 4)
        self.assertEqual(set(report["splits"]["train"]["composition"]), {"aslp/drama", "urdutts/audiobook"})
        self.assertTrue((root / "train.csv").is_file() and (root / "validation.csv").is_file())
        first = report["rows"]["train"][0]["audio"]
        import soundfile as sf
        info = sf.info(root / first)
        self.assertEqual((info.samplerate, info.channels, info.subtype), (24000, 1, "PCM_16"))
        # A new session with deleted audio re-fetches only what is missing and keeps the same identity.
        (root / first).unlink()
        again = self.prepare(self.cfg)
        self.assertEqual(root, again)
        self.assertTrue((root / first).is_file())

    def test_smoke_subset_uses_first_shard_only_and_has_both_sources(self):
        cfg = dict(self.cfg, test_train_clips=4, test_validation_clips=2)
        root = self.prepare(cfg)
        rows = json.loads((root / "dataset_source.json").read_text())["rows"]
        self.assertEqual(len(rows["train"]), 6)  # only the first shard exists in the smoke copy
        self.assertEqual({r["source"] for r in rows["train"]}, {"aslp", "urdutts"})
        self.assertNotEqual(root, self.prepare(self.cfg))  # smoke and full runs never share a directory

    def test_labels_filter_clips_and_add_voice_column(self):
        cfg = dict(self.labelled, work_dir=self.cfg["work_dir"])
        root = self.prepare(cfg)
        report = json.loads((root / "dataset_source.json").read_text())
        self.assertEqual(report["splits"]["train"]["clips"], 9)
        self.assertEqual(report["splits"]["validation"]["clips"], 4)
        self.assertEqual(report["recipe"]["labels"], "labels/v3.jsonl")
        import csv
        with open(root / "train.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["voice"], "spk_0 | female | conversational | happy")
        self.assertNotIn("train-3", {Path(r["audio"]).stem for r in rows})
        self.assertNotEqual(root, self.prepare(self.cfg))  # labelled and plain recipes never share a directory

    def test_labels_and_voice_column_go_together(self):
        with self.assertRaises(ValueError):
            stage2_data.validate_config(dict(self.labelled, speaker_column=None))
        with self.assertRaises(ValueError):
            stage2_data.validate_config(dict(self.cfg, speaker_column="voice"))
        stage2_data.validate_config(self.labelled)

    def test_rejects_unpinned_revision_and_mixture(self):
        with self.assertRaises(ValueError):
            stage2_data.validate_config(dict(self.cfg, dataset_revision="main"))
        with self.assertRaises(ValueError):
            stage2_data.validate_config(dict(self.cfg, mixture={"aslp_fraction": 0.6}))

    def test_epoch_plan_chains_runs_and_changes_seed(self):
        base = json.loads((Path(__file__).resolve().parents[1] / "configs/stage2.json").read_text())
        e2, e3, e4 = (stage2_plan.epoch_config(n, base) for n in (2, 3, 4))
        self.assertIsNone(e2["init_from_run"])
        self.assertEqual(e2["init_source"]["checkpoint"], "checkpoint-5029")
        self.assertEqual(e3["init_from_run"], e2["run_id"])
        self.assertEqual(e4["init_from_run"], e3["run_id"])
        self.assertIsNone(e3["init_source"])
        self.assertEqual(len({e2["seed"], e3["seed"], e4["seed"]}), 3)
        self.assertEqual(len({e2["run_id"], e3["run_id"], e4["run_id"]}), 3)
        self.assertEqual(e2["dataset_revision"], stage2_plan.EPOCH2_DATA["dataset_revision"])
        self.assertEqual((e2["speaker_column"], e3["speaker_column"]), (None, "voice"))
        self.assertEqual(e3["dataset_revision"], e4["dataset_revision"])
        self.assertEqual(base["speaker_column"], "voice")  # base untouched
        with self.assertRaises(ValueError):
            stage2_plan.epoch_config(1, base)


if __name__ == "__main__":
    unittest.main()
