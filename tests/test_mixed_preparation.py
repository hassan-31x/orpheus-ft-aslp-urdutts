import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import mixed_data
from mixture import group_is_validation
from orpheus_utils import sha256


class FixtureScorer:
    def __init__(self, cfg, directory):
        pass

    def acoustic(self, path):
        return dict(dnsmos_ovrl=4., dnsmos_bak=4.5, dnsmos_sig=4., vad_fraction=.9), None

    def alignment(self, path, text):
        return dict(cer=.01, wer=.02, asr_text=text), None


class PreparationTests(unittest.TestCase):
    def test_real_conversion_selection_regeneration_and_source_labels(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/mixed.json").read_text())
        cfg.update(test_train_clips=20, test_validation_clips=10)
        cfg["mixture"]["speaker_max_seconds"] = 1000
        groups = {False: [], True: []}
        for i in range(500):
            group = "podcast/"+str(i)
            groups[group_is_validation(group, .1, 3407)].append(group)
        aslp = []
        payloads = {}

        def audio(i):
            stream = io.BytesIO()
            wave = .1*np.sin(2*np.pi*(190+i*7)*np.arange(72000)/24000)
            sf.write(stream, wave, 24000, format="WAV", subtype="PCM_16")
            return stream.getvalue()

        for i in range(40):
            identifier = f"{i:024x}"
            aslp.append(dict(id=identifier, source="aslp", category="podcast", group=groups[i>=30][i%10],
                text=f"آج ہم آپ سے بات کریں گے {i}", metadata_duration=3., source_path=identifier+".wav", labels={}))
            payloads[identifier+".wav"] = audio(i)

        def tts(cfg, root, scorer, target_seconds, report):
            rows = []
            for i in range(40):
                identifier = f"{i+100:024x}"
                path = root/"screening"/("urdutts-"+identifier+".wav")
                properties = mixed_data.convert_payload(audio(i+100), path)
                rows.append(dict(id=identifier, source="urdutts", category="read_speech",
                    text=f"کل ہم سب بازار جائیں گے {i}", source_shard="train/a.arrow" if i<30 else "eval/a.arrow",
                    source_row=i, source_split="train" if i<30 else "eval", quality=dict(dnsmos_ovrl=4.,dnsmos_bak=4.5,cer=.01,wer=.01),
                    screening_path=path.relative_to(root).as_posix(),sha256=sha256(path),**properties))
            return rows

        def remote(repo, revision, path, missing_ok=False):
            return payloads[path]

        with tempfile.TemporaryDirectory() as d:
            cfg["work_dir"] = d
            with patch.object(mixed_data, "aslp_candidates", return_value=aslp), \
                 patch.object(mixed_data, "remote_bytes", side_effect=remote), \
                 patch.object(mixed_data, "scan_tts", side_effect=tts):
                root = mixed_data.prepare_dataset(cfg, d, scorer_factory=FixtureScorer)
            report = json.loads((root/"dataset_source.json").read_text())
            self.assertEqual({r["source"] for r in report["selected"]["train"]}, {"aslp","urdutts"})
            self.assertEqual(len(report["selected"]["train"]),20)
            self.assertEqual(len(report["selected"]["validation"]),10)
            self.assertFalse(list((root/"screening").glob("*.wav")))
            picked = next(r for r in report["selected"]["train"] if r["source"]=="aslp")
            path = root/picked["audio"]
            path.unlink()
            with patch.object(mixed_data,"remote_bytes",side_effect=remote), \
                 patch.object(mixed_data,"screen_aslp",side_effect=AssertionError("must not select twice")):
                self.assertEqual(mixed_data.prepare_dataset(cfg,d,scorer_factory=FixtureScorer),root)
            self.assertEqual(sha256(path),picked["sha256"])
            # Remote snapshots restore a report, not a persistent Kaggle data directory.
            import shutil
            saved = Path(d)/"snapshot-dataset-source.json"
            saved.write_bytes((root/"dataset_source.json").read_bytes())
            shutil.rmtree(root)
            restored = mixed_data.restore_selection(cfg,d,saved)
            def source_rows(cfg, shard, limit):
                return ((i,dict(audio=dict(bytes=audio(i+100)))) for i in range(limit))
            with patch.object(mixed_data,"remote_bytes",side_effect=remote), \
                 patch.object(mixed_data,"stream_rows",side_effect=source_rows), \
                 patch.object(mixed_data,"screen_aslp",side_effect=AssertionError("do not rerank a restored run")):
                mixed_data.prepare_dataset(cfg,d,scorer_factory=FixtureScorer)
            for rows in report["selected"].values():
                for row in rows:
                    self.assertEqual(sha256(restored/row["audio"]),row["sha256"])


if __name__ == "__main__":
    unittest.main()
