import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orpheus_utils import SPECIAL
from pron_words import (FRAME_SECONDS, ctc_align, frame_multipliers, matched_words, normalize,
                        score_words, word_spans, word_weight)
from weighted_loss import Weighting, validate

SPEC = dict(repo_id="o/r", path="mining/v1/loss_weights.json", revision="a" * 40)


def row(audio, frames, text_tokens=3):
    prefix = [SPECIAL["start_human"]] + [1] * text_tokens + [SPECIAL["start_speech"]]
    ids = prefix + [200000] * (7 * frames) + [SPECIAL["end_speech"], SPECIAL["end_ai"]]
    return dict(input_ids=ids, labels=list(ids), audio=audio)


class WordScoringTest(unittest.TestCase):
    def test_normalize_folds_arabic_letters_and_diacritics(self):
        self.assertEqual(normalize("كيا  بات ہَے!"), normalize("کیا بات ہے"))

    def test_matched_words_marks_substitutions_and_deletions(self):
        self.assertEqual(matched_words("a b c d".split(), "a x c".split()), [True, False, True, False])
        self.assertEqual(matched_words("a b".split(), "a b".split()), [True, True])
        self.assertEqual(matched_words("a b".split(), []), [False, False])

    def test_whisper_errors_on_real_audio_are_not_blamed_on_the_model(self):
        items = [dict(reference="ایک مشکل لفظ", real="ایک مشکل لفظ", synth=["ایک مشکیل لفظ", "ایک مشکل لفظ"]),
                 dict(reference="مشکل کام", real="مشکل قام", synth=["مشکیل قام", "مشکیل کام"])]
        stats = score_words(items)
        hard = stats[normalize("مشکل")]
        self.assertEqual((hard["trials"], hard["errors"], hard["unstable"], hard["unknown"]), (4, 3, 1, 1))
        self.assertEqual(stats[normalize("کام")]["whisper_unreliable"], 1)
        self.assertEqual(stats[normalize("کام")]["trials"], 0)
        self.assertAlmostEqual(hard["weight"], 1 + 2 * 0.75)
        self.assertEqual(stats[normalize("ایک")]["weight"], 1.0)

    def test_word_weight_thresholds(self):
        self.assertEqual(word_weight(1, 1), 1.0)  # too few trials
        self.assertEqual(word_weight(8, 1), 1.0)  # below 25%
        self.assertEqual(word_weight(2, 2), 3.0)  # capped

    def test_shrinkage_needs_evidence_beyond_one_wrong_seed(self):
        prior = dict(min_error_rate=0.3, prior_rate=0.18, prior_strength=4.0)
        self.assertEqual(word_weight(2, 1, **prior), 1.0)  # one wrong seed of two
        self.assertAlmostEqual(word_weight(2, 2, **prior), 1 + 2 * (2 + 0.72) / 6)
        self.assertEqual(word_weight(20, 4, **prior), 1.0)  # 20%: near the corpus rate
        self.assertGreater(word_weight(20, 10, **prior), 1.8)


class AlignmentTest(unittest.TestCase):
    def test_ctc_alignment_recovers_word_spans(self):
        import numpy as np
        blank, a, b, sep = 0, 1, 2, 3
        frames = [blank, a, a, blank, sep, b, b, b, blank]
        lp = np.full((len(frames), 4), -10.0)
        lp[np.arange(len(frames)), frames] = 0.0
        path = ctc_align(lp, [a, sep, b], blank)
        spans = word_spans(path, [0, -1, 1], 0.02)
        self.assertEqual([tuple(round(x, 2) for x in s) for s in spans], [(0.02, 0.06), (0.1, 0.16)])
        self.assertIsNone(ctc_align(lp[:2], [a, sep, b], blank))

    def test_frame_multipliers_cover_span_with_margin(self):
        m = frame_multipliers([[2 * FRAME_SECONDS + 0.01, 3 * FRAME_SECONDS + 0.01, 2.5]], 8, margin_frames=1)
        self.assertEqual(m, [1.0, 2.5, 2.5, 2.5, 2.5, 1.0, 1.0, 1.0])


class WeightingTest(unittest.TestCase):
    def setUp(self):
        mined = dict(spans={"tts1": [[0.0, 0.05, 3.0]]}, aslp_cer={"aslp_bad": 0.9, "aslp_ok": 0.2})
        sources = dict(tts1="urdutts", tts2="urdutts", tts3="urdutts", aslp_ok="aslp", aslp_bad="aslp")
        self.w = Weighting(dict(SPEC, margin_frames=0), mined, sources)

    def test_validate(self):
        validate(None)
        validate(SPEC)
        with self.assertRaises(ValueError):
            validate(dict(SPEC, revision="main"))
        with self.assertRaises(ValueError):
            validate(dict(SPEC, unknown=1))

    def test_per_token_weights_by_source_codebook_and_hard_word(self):
        r = row("audio/train/tts1.flac", 2)
        weights = self.w.raw(r["input_ids"], r["labels"], r["audio"])
        start = r["input_ids"].index(SPECIAL["start_speech"]) + 1
        self.assertEqual(weights[:start], [1.0] * start)
        self.assertEqual([round(x, 6) for x in weights[start:start + 7]], [3.6] + [3.0] * 6)  # hard frame: 1.2 coarse x 3, fine x 3
        self.assertEqual(weights[start + 7:start + 14], [1.2] + [1.0] * 6)
        self.assertEqual(weights[-2:], [1.0, 1.0])
        ok = row("audio/train/aslp_ok.flac", 1)
        self.assertEqual(self.w.raw(ok["input_ids"], ok["labels"], ok["audio"])[start:start + 7], [0.5] + [1.0] * 6)
        bad = row("audio/train/aslp_bad.flac", 1)
        self.assertEqual([round(x, 3) for x in self.w.raw(bad["input_ids"], bad["labels"], bad["audio"])[start:start + 7]],
                         [0.15] + [0.3] * 6)

    def test_calibration_makes_mean_weight_one(self):
        rows = [row("audio/train/tts1.flac", 2), row("audio/train/aslp_ok.flac", 3)]
        self.w.calibrate(rows)
        weights = [w for r in rows for w in self.w(r)]
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)

    def test_oversampling_keeps_length_and_drops_easy_urdutts_only(self):
        audio = ["audio/train/tts1.flac", "audio/train/tts2.flac", "audio/train/tts3.flac", "audio/train/aslp_ok.flac"]
        order, report = self.w.oversample_order(audio, seed=1)
        self.assertEqual(len(order), len(audio))
        self.assertEqual(order.count(0), 2)
        self.assertIn(3, order)
        self.assertEqual(report["dropped_easy"], 1)


class ChunkedLossTest(unittest.TestCase):
    def test_uniform_weights_match_standard_shifted_cross_entropy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        import torch.nn.functional as F
        from weighted_loss import weighted_causal_loss
        torch.manual_seed(0)

        class Body(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(50, 8)

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return (self.embed(input_ids),)

        class LM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model, self.head = Body(), torch.nn.Linear(8, 50)

            def get_output_embeddings(self):
                return self.head

        lm = LM()
        ids = torch.randint(0, 50, (2, 11))
        labels = ids.clone()
        labels[0, :3] = -100
        weights = torch.ones(2, 11)
        total, count = weighted_causal_loss(lm, ids, torch.ones_like(ids), labels, weights, chunk_tokens=4)
        logits = lm.head(lm.model(ids)[0])[:, :-1]
        expected = F.cross_entropy(logits.reshape(-1, 50), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        self.assertAlmostEqual(total.item(), expected.item(), places=4)
        self.assertEqual(count.item(), (labels[:, 1:] != -100).sum().item())
        total.backward()
        self.assertIsNotNone(lm.head.weight.grad)
        weights[1] = 2.0
        doubled, _ = weighted_causal_loss(lm, ids, torch.ones_like(ids), labels, weights, chunk_tokens=4)
        row1 = F.cross_entropy(logits[1], labels[1, 1:], reduction="sum")
        self.assertAlmostEqual(doubled.item(), expected.item() + row1.item(), places=3)


if __name__ == "__main__":
    unittest.main()
