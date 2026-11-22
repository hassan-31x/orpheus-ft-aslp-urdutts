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
LEARNING_RATES = {2: 1e-4, 3: 5e-5, 4: 3e-5, 5: 2e-5, 6: 4e-5, 7: 2e-5, 8: 1e-5}
FIRST_EPOCH = 2
# Every checkpoint is ~2.4 GiB on the Hub (private storage is capped), so from epoch 4 save every 500 steps.
SAVE_STEPS_FROM_EPOCH4 = 500
# Epoch 2 trained on the v2 data with plain-text prompts. From epoch 3, configs/stage2.json selects the v3
# labels: prompts carry "speaker | gender | style | emotion" and acted ASLP dialogue is capped.
# From epoch 5 the loss focuses on pronunciation (weighted_loss.py): hard words mined from the epoch-4
# checkpoint (mine_hard_words.py) are up-weighted in UrduTTS clips; ASLP coarse tokens are down-weighted.
# Pinned after the mining run publishes loss_weights.json; epoch 5 refuses to start until it is set.
# From epoch 6 the data is the stage-3 (v4) mixture: ASLP conversational + UAT read speech + UrduTTS, built by
# stage3_build.py. Epoch 6 restarts at 4e-5 (not the 1e-5 the old plan had) because it sees new vocabulary and a new
# speaker; epochs 7/8 step down again. The hard words are NOT re-mined: the epoch-5 weights stay pinned so
# words the model already says correctly keep being trained. The dataset revision is read at run time from the
# pointer stage3_build.py publishes, so no code change (and no new smoke receipt) sits between the data build and training.
LOSS_WEIGHTING_FROM_EPOCH = 5
DATA_V4_FROM_EPOCH = 6
DATA_V4_POINTER = dict(repo_id="hassan-31x/orpheus-urdu-stage2-checkpoints", path="datasets/stage3-v4.json")
# v4 has almost no UrduTTS clip without a hard word, so "repeat hard clips, drop as many easy ones" would delete the
# easy clips entirely: keep every clip once and rely on the per-token word weights.
V4_EVAL_STEPS = 250  # an eval costs ~2 minutes; 250 keeps ~13 per epoch instead of ~32
# Hard words mined from the epoch-4 final checkpoint (1,168 words, 12,851 UrduTTS clips with aligned spans).
LOSS_WEIGHTING = dict(repo_id="hassan-31x/orpheus-urdu-stage2-checkpoints", path="mining/v1/loss_weights.json",
                      revision="f3e9fee0f656ee9b66fd03b34481aefb6f7c94e6")
EPOCH2_DATA = dict(dataset_revision="18f318476960bde581c6bfec713006c9e211c02a", stage2_labels=None,
                   speaker_column=None, minimum_train_hours=40)


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e").replace("e-", "e")


def run_id(epoch, lr=None):
    lr = LEARNING_RATES[epoch] if lr is None else lr
    return f"stage2-r32-lr{lr_tag(lr)}-seed{BASE_SEED + epoch - FIRST_EPOCH}-epoch{epoch}"


def load_data_pointer(token=None):
    """The dataset revision stage3_build.py published (needs HF_TOKEN read access to the checkpoint repo)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATA_V4_POINTER["repo_id"], DATA_V4_POINTER["path"], token=token)
    return json.loads(Path(path).read_text())


def epoch_config(epoch, base, learning_rate=None, data_pointer=None):
    """Return the config for stage-2 epoch `epoch` derived from configs/stage2.json (never mutates `base`).

    Epochs >= DATA_V4_FROM_EPOCH need `data_pointer` (see load_data_pointer)."""
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
        if epoch >= 4:
            cfg["save_steps"] = SAVE_STEPS_FROM_EPOCH4
        if epoch >= LOSS_WEIGHTING_FROM_EPOCH:
            if not LOSS_WEIGHTING:
                raise ValueError(f"Epoch {epoch} trains with pronunciation loss weights: run mine_hard_words.py and "
                                 "pin LOSS_WEIGHTING in stage2_plan.py first")
            cfg["loss_weighting"] = dict(LOSS_WEIGHTING)
        if epoch >= DATA_V4_FROM_EPOCH:
            if not data_pointer:
                raise ValueError(f"Epoch {epoch} trains on the stage-3 dataset: pass data_pointer (load_data_pointer) "
                                 "after stage3_build.py has published it")
            cfg.update(dataset_id=data_pointer["dataset_id"], dataset_revision=data_pointer["dataset_revision"],
                       stage2_labels=data_pointer["stage2_labels"], eval_steps=V4_EVAL_STEPS,
                       minimum_train_hours=min(cfg["minimum_train_hours"], int(data_pointer["train_hours"] * 0.9)))
            cfg["loss_weighting"] = dict(cfg["loss_weighting"], oversample=1)
    return cfg


def load_base(repo):
    return json.loads((Path(repo) / "configs/stage2.json").read_text())
