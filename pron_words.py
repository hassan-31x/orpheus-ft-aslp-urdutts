"""Word-level pronunciation scoring and CTC forced alignment (pure Python/NumPy, no GPU).

Used by mine_hard_words.py (find the words the model gets wrong), by finetune_mixed.py (turn word
time spans into per-token loss weights) and by eval_pronunciation.py (seed-consistency metrics).
"""
from __future__ import annotations

import re

from orpheus_utils import clean_text

# Whisper sometimes writes Arabic code points for Urdu letters; score both sides on one alphabet.
_FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۀ": "ۂ", "‌": "", "‍": ""})

# SNAC 24 kHz: 512-sample hop, coarse codebook every 4th hop -> one 7-token frame per 2048 samples.
FRAME_SECONDS = 2048 / 24000
TOKENS_PER_FRAME = 7


def normalize(text):
    """NFC, Urdu-folded, no punctuation or diacritics, single spaces (CER counts spaces)."""
    # \w excludes combining marks, so zer/zabar/pesh and similar diacritics are removed here too.
    return clean_text(re.sub(r"[^\w\s]", "", clean_text(text).translate(_FOLD)))


def words(text):
    return normalize(text).split()


def matched_words(reference, hypothesis):
    """For each reference word: True when an optimal edit alignment matches it exactly."""
    ref, hyp = list(reference), list(hypothesis)
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    result, i, j = [False] * n, n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            result[i - 1] = True
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return result


def word_weight(trials, errors, *, min_trials=2, min_error_rate=0.25, scale=2.0, cap=3.0):
    """Loss multiplier for a word from its model-attributable error rate (1.0 = not hard)."""
    if trials < min_trials or not trials:
        return 1.0
    rate = errors / trials
    return 1.0 if rate < min_error_rate else min(cap, 1.0 + scale * rate)


def score_words(items, **weight_options):
    """Model-attributable error statistics per word.

    items: [{"reference": text, "real": whisper text of the real recording, "synth": [whisper text per seed]}]
    A (word occurrence, seed) trial counts only when Whisper heard that word correctly in the REAL
    recording, so Whisper's own mistakes are not blamed on the model.
    """
    stats = {}
    for item in items:
        ref = words(item["reference"])
        real_ok = matched_words(ref, words(item["real"]))
        synth_ok = [matched_words(ref, words(text)) for text in item["synth"]]
        for k, word in enumerate(ref):
            s = stats.setdefault(word, dict(occurrences=0, trials=0, errors=0, unknown=0, unstable=0,
                                            whisper_unreliable=0))
            s["occurrences"] += 1
            if not real_ok[k]:
                s["whisper_unreliable"] += 1
                continue
            outcomes = [ok[k] for ok in synth_ok]
            s["trials"] += len(outcomes)
            s["errors"] += sum(not o for o in outcomes)
            if len(outcomes) > 1:
                s["unknown"] += all(not o for o in outcomes)
                s["unstable"] += any(outcomes) and not all(outcomes)
    for s in stats.values():
        s["weight"] = word_weight(s["trials"], s["errors"], **weight_options)
    return stats


def ctc_align(log_probs, tokens, blank):
    """Viterbi CTC forced alignment. log_probs [T, V] (NumPy), tokens: ids without blanks.

    Returns per-frame token index (-1 for blank) or None when the audio is too short for the text.
    """
    import numpy as np
    T, L = len(log_probs), len(tokens)
    if L == 0 or T < L:
        return None
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = tokens
    S = len(ext)
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    neg = -np.inf
    dp = np.full(S, neg)
    dp[0] = log_probs[0, blank]
    dp[1] = log_probs[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    emit = log_probs[:, ext]
    for t in range(1, T):
        prev1 = np.concatenate(([neg], dp[:-1]))
        prev2 = np.concatenate(([neg, neg], dp[:-2]))
        prev2[~skip] = neg
        stacked = np.stack([dp, prev1, prev2])
        choice = stacked.argmax(axis=0)
        dp = stacked[choice, np.arange(S)] + emit[t]
        back[t] = choice
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if not np.isfinite(dp[s]):
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= back[t, s]
    return np.where(path % 2 == 1, (path - 1) // 2, -1)


def word_spans(frame_tokens, token_word, frame_seconds):
    """(start, end) seconds per word from a frame->token alignment; None for words with no frames.

    token_word maps each aligned token to its word index (-1 for word separators).
    """
    spans = {}
    for t, k in enumerate(frame_tokens):
        if k >= 0 and token_word[k] >= 0:
            w = token_word[k]
            start, end = spans.get(w, (t, t))
            spans[w] = (min(start, t), max(end, t))
    count = max(token_word) + 1 if token_word else 0
    return [(spans[w][0] * frame_seconds, (spans[w][1] + 1) * frame_seconds) if w in spans else None
            for w in range(count)]


def frame_multipliers(spans, frames, margin_frames=1):
    """Max hard-word multiplier for each 7-token SNAC frame. spans: [[start_s, end_s, weight], ...]."""
    result = [1.0] * frames
    for start, end, weight in spans:
        first = max(0, int(start / FRAME_SECONDS) - margin_frames)
        last = min(frames - 1, int(end / FRAME_SECONDS) + margin_frames)
        for f in range(first, last + 1):
            result[f] = max(result[f], weight)
    return result
