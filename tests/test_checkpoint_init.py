import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from checkpoint_init import inspect_source, stage_external_adapter
from orpheus_utils import atomic_json, sha256


class ExternalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/mixed.json").read_text())
        self.api = SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="a"*40))
        self.downloaded = []
        self.run = self.root/"destination"
        self.run.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, complete=True, bad_adapter=False):
        cp = "checkpoint-5000"
        entries = {}
        values = {"status.json": dict(status="complete" if complete else "paused_for_resume"),
                  "resolved_config.json": self.cfg,
                  cp+"/trainer_state.json": dict(global_step=5000 if complete else 3895, max_steps=5000, epoch=1. if complete else .77),
                  cp+"/COMPLETE.json": dict(files={}),
                  cp+"/adapter_config.json": dict(r=32, lora_alpha=64)}
        for name, value in values.items():
            p = self.root/"snapshot"/name
            atomic_json(p, value)
            entries[name] = dict(path="snapshot/"+name, sha256=sha256(p))
        p = self.root/"snapshot"/cp/"adapter_model.safetensors"
        p.write_bytes(b"trained lora tensors")
        entries[cp+"/adapter_model.safetensors"] = dict(path="snapshot/"+cp+"/adapter_model.safetensors", sha256="bad" if bad_adapter else sha256(p))
        seal = self.root/"snapshot"/cp/"COMPLETE.json"
        sealed = {name: entries[cp+"/"+name]["sha256"] for name in
                  ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json")}
        sealed.update({n: "unused-optimizer-digest" for n in ("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth")})
        atomic_json(seal, dict(files=sealed))
        entries[cp+"/COMPLETE.json"]["sha256"] = sha256(seal)
        atomic_json(self.root/"latest.json", dict(format="files-v1", checkpoint=cp, files=entries))

    def download(self, repo, path, revision):
        self.assertEqual(revision, "a"*40)
        relative = path.split("/", 2)[2]
        self.downloaded.append(relative)
        return self.root/relative

    def test_paused_source_is_reported_and_refused(self):
        self.fixture(complete=False)
        info, _ = inspect_source(self.cfg["init_source"], api=self.api, download=self.download)
        self.assertFalse(info["complete"])
        with self.assertRaisesRegex(RuntimeError, "finish epoch 1"):
            stage_external_adapter(self.cfg, self.run, api=self.api, download=self.download)
        self.assertFalse((self.run/"init_adapter").exists())

    def test_import_is_pinned_checked_and_downloads_no_optimizer(self):
        self.fixture()
        result = stage_external_adapter(self.cfg, self.run, api=self.api, download=self.download)
        self.assertEqual((result/"adapter_model.safetensors").read_bytes(), b"trained lora tensors")
        self.assertFalse(any("optimizer" in name for name in self.downloaded))
        self.assertEqual(json.loads((self.run/"init_source.json").read_text())["revision"], "a"*40)
        self.downloaded.clear()
        self.assertEqual(stage_external_adapter(self.cfg, self.run, api=self.api, download=self.download), result)
        self.assertFalse(self.downloaded)

    def test_corrupt_remote_adapter_never_initializes_training(self):
        self.fixture(bad_adapter=True)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            stage_external_adapter(self.cfg, self.run, api=self.api, download=self.download)
        self.assertFalse((self.run/"init_source.json").exists())

    def test_cached_adapter_tampering_is_detected(self):
        self.fixture()
        stage_external_adapter(self.cfg, self.run, api=self.api, download=self.download)
        (self.run/"init_adapter/adapter_model.safetensors").write_bytes(b"corruption")
        with self.assertRaisesRegex(RuntimeError, "corrupt"):
            stage_external_adapter(self.cfg, self.run, api=self.api, download=self.download)

    def test_model_revision_mismatch_is_refused(self):
        self.fixture()
        self.cfg["model_revision"] = "b"*40
        with self.assertRaisesRegex(RuntimeError, "model_revision"):
            stage_external_adapter(self.cfg, self.run, api=self.api, download=self.download)
