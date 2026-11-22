import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stage3_build as s3
from stage3_build import V4

GOOD = dict(dnsmos_sig=3.4, dnsmos_bak=3.7, dnsmos_ovrl=3.1, vad_fraction=0.95, rms_db=-20.0, peak=0.5,
            clipped_fraction=0.0)
RULES = dict(V4, target_train_hours=0.3, minimum_train_hours=0.05, maximum_train_hours=2.0, min_urdutts_hours=0.02,
             uat_validation_clips=4, uat_eval_prompts=3)
VOCAB = ("کتاب قلم دروازہ کھڑکی بازار مسجد باغ پہاڑ دریا سمندر بادل ہوا روشنی خوشبو آواز رنگ پھول پرندہ"
         " ستارہ چاند سورج راستہ گاؤں شہر مکان دکان مدرسہ استاد طالب علم موسم").split()


def text(i, n=0):
    """A distinct, plausible-length Urdu string per index (selection de-duplicates by text)."""
    a, b, c = VOCAB[i % 30], VOCAB[(i // 30) % 30], VOCAB[(i // 900) % 30]
    return f"آج {a} اور {b} کے ساتھ {c} بہت خوشگوار ہے"


def uat_row(i, **extra):
    row = dict(id=f"uat-lj{i:05d}", source="uat", pool_split="train", category="narration", speaker="uat_narrator",
               text=text(i), duration=4.0, metadata_duration=4.0, wav_sha256=f"u{i}", parent_sha256=None,
               provenance="{}", error=None, **GOOD)
    row.update(extra)
    return row


def align_row(i, nll=1.0, cer=0.1, ok=True, unk=0.0):
    return dict(id=f"uat-lj{i:05d}", align_nll=nll, align_cer=cer, unk_fraction=unk, asr_text="", align_ok=ok)


def aslp_row(i, category="podcast", speaker=None):
    return dict(id=f"aslp{i:04d}", source="aslp", pool_split="train", category=category,
                speaker=speaker or f"aslp-{category}-SPEAKER_{i % 6:04d}", text=text(100 + i), duration=5.0,
                metadata_duration=5.0, wav_sha256=f"a{i}", parent_sha256=None, error=None,
                provenance=json.dumps(dict(gender="female", emotion="calm and conversational")), **GOOD)


def tts_row(i, split="train"):
    return dict(id=f"tts-{i:05d}" + ("-val" if split == "validation" else ""), source="urdutts", pool_split=split,
                category="audiobook", speaker=None, text=text(200 + i), duration=6.0, metadata_duration=6.0,
                wav_sha256=f"t{i}{split}", parent_sha256=f"t{i}{split}", provenance="{}", error=None,
                **dict(GOOD, dnsmos_ovrl=3.0 + (i % 10) / 100))


class UatGateTest(unittest.TestCase):
    def test_good_clip_passes_and_bad_ones_name_their_reason(self):
        self.assertIsNone(s3.uat_gate(uat_row(1), RULES))
        self.assertEqual(s3.uat_gate(uat_row(1, duration=0.9), RULES), "duration")
        self.assertEqual(s3.uat_gate(uat_row(1, text="hello ہے"), RULES), "latin_or_empty_text")
        self.assertEqual(s3.uat_gate(uat_row(1, dnsmos_ovrl=2.0), RULES), "dnsmos_below_tier")
        self.assertEqual(s3.uat_gate(uat_row(1, error="convert"), RULES), "error")
        self.assertEqual(s3.uat_gate(uat_row(1, rms_db=-60.0), RULES), "loudness")

    def test_text_is_nfc_and_whitespace_collapsed(self):
        self.assertEqual(s3.clean_uat_text("  آپ   کیسے\tہیں \n"), "آپ کیسے ہیں")


class ChooseUatTest(unittest.TestCase):
    def pool(self, n=100):
        uat = [uat_row(i) for i in range(n)]
        align = {r["id"]: align_row(i, nll=1.0 + i / 100, cer=0.05 + (i % 10) / 100) for i, r in enumerate(uat)}
        return uat, align

    def test_worst_aligned_clips_are_dropped_and_validation_is_the_cleanest(self):
        uat, align = self.pool()
        train, val, report = s3.choose_uat(uat, align, RULES, random.Random(1), set())
        self.assertEqual(len(val), RULES["uat_validation_clips"])
        kept = len(train) + len(val)
        self.assertLessEqual(kept, 85)  # 15% worst by alignment score dropped
        self.assertGreaterEqual(kept, 80)
        median = s3.quantile([align[r["id"]]["align_nll"] for r in train + val], 0.5)
        self.assertTrue(all(align[r["id"]]["align_nll"] <= median for r in val))
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in val})
        self.assertEqual(report["rejected"]["text_audio_mismatch"], 100 - kept)

    def test_rejects_missing_alignment_unalignable_text_and_far_off_transcripts(self):
        uat, align = self.pool(20)
        del align[uat[0]["id"]]
        align[uat[1]["id"]] = align_row(1, ok=False)
        align[uat[2]["id"]] = align_row(2, unk=0.5)
        align[uat[3]["id"]] = align_row(3, cer=0.9)
        _, _, report = s3.choose_uat(uat, align, RULES, random.Random(1), set())
        self.assertEqual(report["rejected"]["no_alignment"], 1)
        self.assertEqual(report["rejected"]["unalignable_text"], 2)
        self.assertGreaterEqual(report["rejected"]["text_audio_mismatch"], 1)

    def test_texts_from_other_sources_and_duplicates_are_removed(self):
        uat, align = self.pool(10)
        uat[5]["text"] = uat[4]["text"]  # duplicate text
        uat[6]["wav_sha256"] = uat[7]["wav_sha256"]  # duplicate audio
        banned = {s3.b2.norm_text(uat[8]["text"])}
        _, _, report = s3.choose_uat(uat, align, RULES, random.Random(1), banned)
        self.assertEqual(report["rejected"]["duplicate"], 2)
        self.assertEqual(report["rejected"]["text_in_other_source"], 1)


class PickUrduTtsTest(unittest.TestCase):
    def test_hard_word_clips_come_first_then_cleanest_easy_clips(self):
        rows = [tts_row(i) for i in range(40)]
        spans = {rows[i]["id"]: [[0.1, 0.5, 2.0]] for i in (30, 31, 32, 33)}
        spans[rows[33]["id"]] = [[0.1, 0.5, 3.0], [1.0, 1.4, 3.0]]  # most pronunciation signal
        hours = 24 * 6 / 3600  # 24 clips
        picked = s3.pick_urdutts(rows, spans, hours, RULES)
        ids = [r["id"] for r in picked]
        self.assertEqual(len(picked), 24)
        self.assertEqual(ids[0], rows[33]["id"])
        self.assertTrue({rows[i]["id"] for i in (30, 31, 32, 33)} <= set(ids))
        self.assertEqual(len(set(ids)), 24)


class SelectV4Test(unittest.TestCase):
    def pools(self):
        aslp = [aslp_row(i, category=c) for i, c in enumerate(["podcast"] * 24 + ["drama"] * 24 + ["interview"] * 12)]
        tts = [tts_row(i) for i in range(120)] + [tts_row(i, "validation") for i in range(30)]
        uat = [uat_row(i) for i in range(60)]
        align = {r["id"]: align_row(i, nll=1 + i / 60) for i, r in enumerate(uat)}
        spans = {tts[i]["id"]: [[0.2, 0.6, 2.5]] for i in range(0, 120, 4)}
        return aslp, tts, uat, align, spans

    def test_selection_is_leak_free_and_labelled(self):
        train, val, report = s3.select_v4(*self.pools(), rules=RULES)
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in val})
        self.assertEqual({r["source"] for r in train}, {"aslp", "uat", "urdutts"})
        self.assertEqual({r["source"] for r in val} - {"aslp"}, {"uat", "urdutts"})
        self.assertEqual(report["train"]["clips"], len(train))
        self.assertGreater(report["train"]["conversational_fraction"], 0)
        labels = s3.make_labels(train, val)
        self.assertEqual(len(labels), len(train) + len(val))
        voices = {r["id"]: r["voice"] for r in labels}
        self.assertTrue(all(voices[r["id"]] == s3.UAT_VOICE for r in train if r["source"] == "uat"))
        self.assertTrue(all(voices[r["id"]].startswith("urdutts_narrator") for r in train if r["source"] == "urdutts"))
        self.assertTrue(all("| conversational |" in voices[r["id"]] for r in train if r["source"] == "aslp"))
        self.assertTrue(all(r["keep"] for r in labels))

    def test_validation_clips_for_urdutts_and_aslp_match_the_stage2_build(self):
        # Same RNG call order as build_stage2.phase_select keeps held-out clips comparable across epochs.
        aslp, tts, uat, align, spans = self.pools()
        _, val, _ = s3.select_v4(aslp, tts, uat, align, spans, rules=RULES)
        rng = random.Random(s3.b2.RULES["seed"])
        _, aslp_val, _, _, _ = s3.b2.choose_aslp(aslp, rng)
        pool = sorted((r for r in tts if r["pool_split"] == "validation"), key=lambda r: r["id"])
        rng.shuffle(pool)
        self.assertEqual({r["id"] for r in val if r["source"] != "uat"},
                         {r["id"] for r in aslp_val} | {r["id"] for r in pool[:250]})

    def test_out_of_range_hours_are_reported(self):
        _, _, report = s3.select_v4(*self.pools(), rules=dict(RULES, minimum_train_hours=50.0))
        self.assertEqual(report["status"], "out_of_range")

    def test_default_voice_candidates_are_neutral_non_acted_speakers(self):
        train, _, _ = s3.select_v4(*self.pools(), rules=RULES)
        for r in train:
            if r["source"] == "aslp":
                r["provenance"] = json.dumps(dict(gender="male", emotion="calm neutral"))
        voices = s3.conversational_voices(train, minimum_minutes=0.1)
        self.assertTrue(voices)
        self.assertTrue(all(v["voice"].endswith("| neutral") and "drama" not in v["voice"] for v in voices))


class VerifyFinalTest(unittest.TestCase):
    def build(self, root, drop_label=False, rate=24000):
        try:
            import io
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile/pyarrow not installed")
        root = Path(root)
        rows = {"train": [uat_row(1), aslp_row(2), tts_row(3)], "validation": [uat_row(4), tts_row(5, "validation")]}
        labels = []
        for split, items in rows.items():
            (root / split).mkdir(parents=True)
            meta = [c for c in s3.b2.POOL_COLUMNS if c != "flac"]
            audio = []
            for r in items:
                buffer = io.BytesIO()
                sf.write(buffer, (0.1 * np.sin(np.arange(int(r["duration"] * rate)) * 0.05)).astype("float32"), rate,
                         format="FLAC", subtype="PCM_16")
                audio.append(dict(bytes=buffer.getvalue(), path=r["id"] + ".flac"))
            pq.write_table(pa.table({"audio": audio, **{c: [r.get(c) for r in items] for c in meta}}),
                           root / split / f"{split}-00000-of-00001.parquet")
            (root / "manifests").mkdir(exist_ok=True)
            with open(root / "manifests" / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for r in items:
                    f.write(json.dumps({c: r.get(c) for c in meta}, ensure_ascii=False) + "\n")
            labels += s3.make_labels(items if split == "train" else [], items if split == "validation" else [])
        (root / "labels").mkdir()
        with open(root / "labels/v4.jsonl", "w", encoding="utf-8") as f:
            for r in labels[1:] if drop_label else labels:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_consistent_dataset_passes_and_reports_hours(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.build(d)
            hours = s3.verify_final(d)
        self.assertEqual(set(hours), {"train/uat", "train/aslp", "train/urdutts", "validation/uat", "validation/urdutts"})

    def test_missing_label_or_wrong_sample_rate_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.build(d, drop_label=True)
            with self.assertRaisesRegex(RuntimeError, "label"):
                s3.verify_final(d)
        with tempfile.TemporaryDirectory() as d:
            self.build(d, rate=16000)
            with self.assertRaisesRegex(RuntimeError, "Hz"):
                s3.verify_final(d)


class SelectPhaseTest(unittest.TestCase):
    """phase_select end to end on small synthetic pools: reads parts, selects, writes shards, verifies them."""

    def test_writes_a_verified_dataset(self):
        try:
            import io
            import types
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
            import soundfile as sf
            import tempfile
        except ImportError:
            self.skipTest("soundfile/pyarrow not installed")

        def flac(seconds, seed):
            buffer = io.BytesIO()
            sf.write(buffer, (0.1 * np.sin(np.arange(int(seconds * 24000)) * (0.01 + 0.001 * seed))).astype("float32"),
                     24000, format="FLAC", subtype="PCM_16")
            return buffer.getvalue()

        aslp = [aslp_row(i, category=c) for i, c in enumerate(["podcast"] * 24 + ["drama"] * 24 + ["interview"] * 12)]
        tts = [tts_row(i) for i in range(120)] + [tts_row(i, "validation") for i in range(30)]
        uat = [uat_row(i) for i in range(60)]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            for prefix, rows in (("pool/aslp", aslp), ("pool/urdutts", tts)):
                (source / prefix).mkdir(parents=True)
                for k in range(0, len(rows), 40):
                    s3.b2.write_part([dict(r, flac=flac(r["duration"], j)) for j, r in enumerate(rows[k:k + 40])],
                                     source / prefix / f"part-{k // 40:05d}.parquet")
            work = root / "work"
            (work / "pool/uat").mkdir(parents=True)
            s3.b2.write_part([dict(r, flac=flac(r["duration"], j)) for j, r in enumerate(uat)],
                             work / "pool/uat/part-00000.parquet")
            (work / "pool/uat-align").mkdir()
            align_rows = [align_row(i, nll=1 + i / 60) for i in range(60)]
            pq.write_table(pa.table({c: [r[c] for r in align_rows] for c in s3.ALIGN_COLUMNS}),
                           work / "pool/uat-align/part-00000.parquet")

            class LocalPool:
                enabled = True

                def __init__(self, *args, **kwargs):
                    pass

                def names(self, prefix=""):
                    return sorted(str(p.relative_to(source)) for p in source.rglob("*.parquet")
                                  if str(p.relative_to(source)).startswith(prefix))

                def get(self, remote, local_dir):
                    target = Path(local_dir) / remote
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((source / remote).read_bytes())
                    return str(target)

            spans = {tts[i]["id"]: [[0.2, 0.6, 2.5]] for i in range(0, 120, 4)}
            args = types.SimpleNamespace(work=str(work), token="", mining=dict(spans=spans), pool_parts=0,
                                         no_hub=True, allow_out_of_range=True)
            original = (s3.PoolSource, s3.V4)
            s3.PoolSource = LocalPool
            try:
                out = s3.phase_select(args, None)
            finally:
                s3.PoolSource = original[0]
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["train"]["clips"] + report["validation"]["clips"],
                             len(s3.make_labels([], [])) + sum(1 for _ in (out / "labels/v4.jsonl").read_text().splitlines()))
            self.assertTrue(list((out / "train").glob("*.parquet")) and list((out / "validation").glob("*.parquet")))
            self.assertTrue((out / "eval/uat_heldout_prompts.csv").is_file())
            self.assertIn("conversational_voice_candidates", report)


class CtcScoresTest(unittest.TestCase):
    def test_matching_transcript_scores_lower_nll_and_greedy_decode_reads_it_back(self):
        try:
            import numpy as np
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        blank, vocab = 0, 5
        path = [1, 1, 0, 2, 2, 0, 3, 0, 4, 4]
        logits = np.full((len(path), vocab), -8.0, dtype="float32")
        for t, k in enumerate(path):
            logits[t, k] = 0.0
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        chars = {1: "ا", 2: "ب", 3: "پ", 4: "ت", 0: ""}
        good, greedy = s3.ctc_scores(log_probs, [1, 2, 3, 4], blank, chars)
        bad, _ = s3.ctc_scores(log_probs, [4, 3, 2, 1], blank, chars)
        self.assertLess(good, 0.1)
        self.assertGreater(bad, good + 5)
        self.assertEqual(greedy, "ابپت")
        impossible, _ = s3.ctc_scores(log_probs[:3], [1, 2, 3, 4], blank, chars)
        self.assertEqual(impossible, float("inf"))


if __name__ == "__main__":
    unittest.main()
