"""Pronunciation-focused loss weighting for stage-2 epochs (epoch 5 onward).

Weights come from a mining file (mine_hard_words.py) pinned at a Hub commit:
  * UrduTTS clips teach pronunciation: audio tokens inside words the previous epoch mispronounced
    get their word's multiplier (1-3x), and coarse SNAC tokens (frame position 0) get `tts_coarse`.
  * ASLP clips teach natural delivery: coarse tokens get `aslp_coarse` (<1) so noisy conversational
    transcripts steer pronunciation less, and clips whose transcript Whisper could not confirm get
    `aslp_unreliable` on every audio token.
  * UrduTTS clips that contain hard words are repeated `oversample` times; the same number of
    UrduTTS clips without hard words are dropped, so epoch length and source balance stay fixed.
All weights are divided by their training-set mean so the effective learning rate is unchanged.
Validation loss stays unweighted, so it remains comparable with earlier epochs.
"""
from __future__ import annotations

import random
from pathlib import Path

from orpheus_utils import SPECIAL, read_json
from pron_words import TOKENS_PER_FRAME, frame_multipliers

DEFAULTS = dict(tts_coarse=1.2, tts_fine=1.0, aslp_coarse=0.5, aslp_fine=1.0, aslp_unreliable=0.3,
                aslp_unreliable_cer=0.6, margin_frames=1, oversample=2, chunk_tokens=512)
REQUIRED = ("repo_id", "path", "revision")


def validate(spec):
    if spec is None:
        return
    missing = [k for k in REQUIRED if not spec.get(k)]
    if missing:
        raise ValueError(f"loss_weighting needs {missing}")
    if len(spec["revision"]) != 40:
        raise ValueError("loss_weighting.revision must be a full 40-character Hub commit")
    unknown = set(spec) - set(REQUIRED) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown loss_weighting fields {sorted(unknown)}")
    options = {**DEFAULTS, **spec}
    if any(options[k] <= 0 for k in DEFAULTS if k not in ("margin_frames", "oversample")):
        raise ValueError("loss_weighting multipliers must be positive")
    if options["margin_frames"] < 0 or options["oversample"] < 1:
        raise ValueError("margin_frames >= 0 and oversample >= 1")


def load(spec, token=None):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(spec["repo_id"], spec["path"], revision=spec["revision"], token=token)
    return read_json(Path(path))


class Weighting:
    def __init__(self, spec, mined, sources):
        """sources: clip id -> "aslp" | "urdutts" (training manifest)."""
        validate(spec)
        self.o = {**DEFAULTS, **spec}
        self.spans = mined.get("spans", {})
        self.aslp_cer = mined.get("aslp_cer", {})
        self.sources = sources
        self.scale = 1.0

    @staticmethod
    def clip_id(audio):
        return Path(audio).stem

    def raw(self, input_ids, labels, audio):
        """Unnormalised per-label weights for one encoded row (same length as labels)."""
        weights = [1.0] * len(labels)
        clip = self.clip_id(audio)
        source = self.sources.get(clip)
        if source not in ("aslp", "urdutts"):
            return weights
        try:
            start = list(input_ids).index(SPECIAL["start_speech"]) + 1
            end = list(input_ids).index(SPECIAL["end_speech"], start)
        except ValueError:
            return weights
        frames = (end - start) // TOKENS_PER_FRAME
        if source == "urdutts":
            coarse, fine = self.o["tts_coarse"], self.o["tts_fine"]
            multiplier = frame_multipliers(self.spans.get(clip, []), frames, self.o["margin_frames"])
        else:
            coarse, fine = self.o["aslp_coarse"], self.o["aslp_fine"]
            cer = self.aslp_cer.get(clip)
            reliability = self.o["aslp_unreliable"] if cer is not None and cer > self.o["aslp_unreliable_cer"] else 1.0
            coarse, fine = coarse * reliability, fine * reliability
            multiplier = [1.0] * frames
        for k in range(frames * TOKENS_PER_FRAME):
            if labels[start + k] != -100:
                weights[start + k] = (coarse if k % TOKENS_PER_FRAME == 0 else fine) * multiplier[k // TOKENS_PER_FRAME]
        return weights

    def __call__(self, row):
        return [w / self.scale for w in self.raw(row["input_ids"], row["labels"], row["audio"])]

    def calibrate(self, dataset):
        """Mean raw weight over every trained label token in the (oversampled) training set -> 1."""
        total = count = 0.0
        for row in dataset:
            labels = row["labels"]
            for w, y in zip(self.raw(row["input_ids"], labels, row["audio"]), labels):
                if y != -100:
                    total += w
                    count += 1
        self.scale = total / count if count else 1.0
        return self.scale

    def oversample_order(self, audio_column, seed):
        """Row indices: hard UrduTTS clips repeated, an equal number of easy UrduTTS clips dropped."""
        hard, easy, other = [], [], []
        for i, audio in enumerate(audio_column):
            clip = self.clip_id(audio)
            if self.sources.get(clip) != "urdutts":
                other.append(i)
            elif self.spans.get(clip):
                hard.append(i)
            else:
                easy.append(i)
        rng = random.Random(seed)
        extra = (self.o["oversample"] - 1) * len(hard)
        extra = min(extra, len(easy))  # never drop more easy clips than exist
        rng.shuffle(easy)
        repeats = [hard[i % len(hard)] for i in range(extra)] if hard else []
        order = other + hard + repeats + easy[extra:]
        report = dict(hard_urdutts=len(hard), easy_urdutts=len(easy), other=len(other),
                      repeated=len(repeats), dropped_easy=extra, rows=len(order))
        return sorted(order), report


def weighted_causal_loss(model, input_ids, attention_mask, labels, weights, chunk_tokens=512):
    """Sum over tokens of weight * cross-entropy, computed in checkpointed chunks.

    The full [sequence, 156k-vocabulary] logits never exist at once: each chunk's logits are
    recomputed during backward. Returns (weighted_sum, label_token_count).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)[0]
    hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    target = labels[:, 1:].reshape(-1)
    weight = weights[:, 1:].reshape(-1).to(torch.float32)
    head = inner.get_output_embeddings()

    def chunk(h, y, w):
        logits = head(h).float()
        return (F.cross_entropy(logits, y, ignore_index=-100, reduction="none") * w).sum()

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, target.numel(), chunk_tokens):
        y = target[i:i + chunk_tokens]
        if (y != -100).any():
            total = total + checkpoint(chunk, hidden[i:i + chunk_tokens], y, weight[i:i + chunk_tokens],
                                       use_reentrant=False)
    return total, (target != -100).sum()
