import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mixture import best_with_budget, curriculum_order, group_is_validation, select_aslp
from mixed_data import jsonl_metadata, validate_config, write_manifest
from quality import metadata_problem, text_problem, transcript_errors


def clip(i, source="aslp", category="podcast", duration=5, score=4.):
    return dict(id=str(i), source=source, category=category, group=str(i//5), duration_seconds=duration,
                quality=dict(dnsmos_ovrl=score, dnsmos_bak=4.5, cer=.02, wer=.04))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/mixed.json").read_text())
        self.mix = self.cfg["mixture"]

    def test_curriculum_visits_every_clip_once_and_keeps_replay_in_every_stage(self):
        rows = [clip(i) for i in range(600)] + [clip(i+600, source="urdutts") for i in range(400)]
        order, report = curriculum_order(rows, 3407)
        self.assertEqual(sorted(order), list(range(1000)))
        self.assertEqual(order, curriculum_order(rows, 3407)[0])
        self.assertNotEqual(order, curriculum_order(rows, 1)[0])
        for actual, target in zip(report, [.5, .6, .7]):
            self.assertAlmostEqual(actual["actual_aslp_fraction"], target, delta=.015)
        # An interrupted Trainer skips the same prefix after restoring its state.
        self.assertEqual(order[400:], curriculum_order(rows, 3407)[0][400:])

    def test_hour_budgets_handle_unequal_clip_lengths_without_upsampling(self):
        rows = [clip(i, duration=17 if i%2 else 2) for i in range(100)]
        chosen = best_with_budget(rows, 100)
        self.assertLessEqual(sum(r["duration_seconds"] for r in chosen), 100)
        self.assertEqual(len(chosen), len({r["id"] for r in chosen}))

    def test_core_and_fallback_caps_hold_even_when_drama_is_plentiful_and_scores_higher(self):
        rows = [clip(i) for i in range(100)]
        rows += [clip(i+100, category="drama", score=4.8) for i in range(300)]
        rows += [clip(i+400, category="roadside_interview") for i in range(200)]
        rows += [clip(i+600, category="news") for i in range(100)]
        chosen = select_aslp(rows, self.mix)
        total = sum(r["duration_seconds"] for r in chosen)
        core = sum(r["duration_seconds"] for r in chosen if r["category"] in ("podcast", "interview"))
        self.assertGreaterEqual(core/total, self.mix["minimum_core_fraction"])
        for cat, cap in self.mix["category_caps"].items():
            self.assertLessEqual(sum(r["duration_seconds"] for r in chosen if r["category"] == cat)/total, cap+1e-9)
        self.assertLessEqual(total/3600, self.mix["max_aslp_hours"])

    def test_drama_cannot_substitute_for_missing_conversation(self):
        with self.assertRaisesRegex(ValueError, "core"):
            select_aslp([clip(i, category="drama") for i in range(100)], self.mix)

    def test_per_speaker_limit(self):
        rows = [dict(clip(i), group="same") for i in range(1000)]
        chosen = best_with_budget(rows, 10000, speaker_cap=180)
        self.assertLessEqual(sum(r["duration_seconds"] for r in chosen), 180)

    def test_holdout_assignment_is_group_consistent_and_seeded(self):
        values = [group_is_validation(str(i), .1, 3407) for i in range(1000)]
        self.assertGreater(sum(values), 70)
        self.assertLess(sum(values), 140)
        self.assertEqual(values, [group_is_validation(str(i), .1, 3407) for i in range(1000)])

    def test_roadside_audio_id_collision_does_not_join_unrelated_clips(self):
        rows = [dict(Audio_Clip="a.wav", audio_id="ROADSIDE_INTERVIEW", Transcription="الف"),
                dict(Audio_Clip="b.wav", audio_id="ROADSIDE_INTERVIEW", Transcription="ب")]
        payload = "\n".join(json.dumps(r) for r in rows).encode()
        self.assertEqual(set(jsonl_metadata(payload, "fixture")), {"a.wav", "b.wav"})
        rows.append(dict(rows[0], Transcription="conflicting"))
        payload = "\n".join(json.dumps(r) for r in rows).encode()
        self.assertEqual(set(jsonl_metadata(payload, "fixture")), {"b.wav"})

    def test_unsafe_metadata_path_refused(self):
        with self.assertRaises(ValueError):
            jsonl_metadata(b'{"Audio_Clip":"../outside.wav"}', "fixture")

    def test_script_and_style_rejections_keep_original_spoken_text(self):
        row = dict(Transcription="آج ہم آپ سے بات کریں گے", Duration_seconds="4.2", Confidence_score="0.98")
        para = dict(accent="Standard Pakistani Urdu.", pronunciation="clear and articulate", contextual="An informal interview")
        self.assertIsNone(metadata_problem(row, para, self.mix))
        for extra in [dict(accent="Standard Pakistani Urdu with Punjabi influence"),
                      dict(contextual="This utterance is sung"), dict(contextual="A poetry recitation"),
                      dict(paralinguistic="Multiple speakers talking")]:
            self.assertIsNotNone(metadata_problem(row, dict(para, **extra), self.mix))
        self.assertIsNotNone(text_problem("آج hello ہم are بات کریں گے", self.mix))
        self.assertEqual(row["Transcription"], "آج ہم آپ سے بات کریں گے")

    def test_asr_errors_tolerate_punctuation_and_urdu_arabic_variants(self):
        self.assertEqual(transcript_errors("یہ کیا ہے؟", "یہ كيا ہے"), dict(cer=0., wer=0.))
        self.assertGreater(transcript_errors("یہ کیا ہے", "")["cer"], .9)
        self.assertGreater(transcript_errors("یہ کیا ہے", "غلط الفاظ")["wer"], .3)

    def test_nonfinite_confidence_is_not_high_quality(self):
        row = dict(Transcription="آج ہم آپ سے بات کریں گے", Duration_seconds=4, Confidence_score="nan")
        self.assertEqual(metadata_problem(row, {}, self.mix), "transcription_confidence")

    def test_full_run_cannot_silently_scan_only_first_urdu_shard(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["mixture"]["tts_scan_limit"] = 20
        with self.assertRaisesRegex(ValueError, "Full runs"):
            validate_config(cfg)

    def test_source_columns_survive_manifest_conversion(self):
        import csv
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"train.csv"
            write_manifest(path, [dict(audio="audio/a.wav", text="یہ اردو ہے", source="aslp", category="podcast")])
            with path.open() as f:
                self.assertEqual(next(csv.DictReader(f))["source"], "aslp")


if __name__ == "__main__":
    unittest.main()
