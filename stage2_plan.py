"""One place that defines every stage-2 epoch run, so epochs 2, 3, 4... are configured identically.

Each epoch is its own run (fresh optimizer/schedule, new shuffle seed) inside one Hub repo:
  epoch 2  starts from the finished UrduTTS epoch-1 adapter (`init_source`)
  epoch N  starts from the finished run of epoch N-1 (`init_from_run`) and reuses its token cache
An interrupted epoch resumes from its own latest verified checkpoint, on any Kaggle account that has
the same HF token: use the same epoch number and the same code commit.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_SEED = 3407
# Warm-restart cosine per epoch; later epochs step the peak LR down to consolidate rather than overwrite.
# Epoch 3 dropped from the planned 7e-5 to 5e-5: epoch 2 at 1e-4 pulled the voice away from clean reading.
LEARNING_RATES = {2: 1e-4, 3: 5e-5, 4: 3e-5, 5: 2e-5, 6: 1e-5}
FIRST_EPOCH = 2
# Epoch 2 trained on the v2 data with plain-text prompts. From epoch 3, configs/stage2.json selects the v3
# labels: prompts carry "speaker | gender | style | emotion" and acted ASLP dialogue is capped.
EPOCH2_DATA = dict(dataset_revision="18f318476960bde581c6bfec713006c9e211c02a", stage2_labels=None,
                   speaker_column=None, minimum_train_hours=40)


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e").replace("e-", "e")


def run_id(epoch, lr=None):
    lr = LEARNING_RATES[epoch] if lr is None else lr
    return f"stage2-r32-lr{lr_tag(lr)}-seed{BASE_SEED + epoch - FIRST_EPOCH}-epoch{epoch}"


def epoch_config(epoch, base, learning_rate=None):
    """Return the config for stage-2 epoch `epoch` derived from configs/stage2.json (never mutates `base`)."""
    if epoch < FIRST_EPOCH:
        raise ValueError(f"Stage-2 epochs start at {FIRST_EPOCH}; epoch 1 was the UrduTTS-only run")
    if epoch not in LEARNING_RATES and learning_rate is None:
        raise ValueError(f"No learning rate defined for epoch {epoch}; pass learning_rate or extend LEARNING_RATES")
    lr = LEARNING_RATES[epoch] if learning_rate is None else learning_rate
    cfg = json.loads(json.dumps(base))
    cfg.update(run_id=run_id(epoch, lr), learning_rate=lr, seed=BASE_SEED + epoch - FIRST_EPOCH)
    if epoch == FIRST_EPOCH:
        cfg.update(EPOCH2_DATA)
        cfg["init_from_run"] = None
    else:
        previous = run_id(epoch - 1, LEARNING_RATES.get(epoch - 1))
        cfg["init_from_run"] = previous
        cfg["init_source"] = None
    return cfg


def load_base(repo):
    return json.loads((Path(repo) / "configs/stage2.json").read_text())
