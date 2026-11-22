#!/usr/bin/env python3
"""Generate the two stage-3 Kaggle notebooks from the proven stage-2 notebook, so they share its bootstrap,
credential, setup and training cells instead of re-implementing them.

  kaggle_stage3_data.ipynb   build + publish the v4 dataset (smoke test first, then the full build)
  kaggle_stage3_train.ipynb  smoke test, then the full epoch, in one run (no manual TEST=True / TEST=False step)

Run `python make_stage3_notebooks.py` after changing the stage-2 notebook; commit the generated notebooks.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "kaggle_stage2.ipynb").read_text(encoding="utf-8"))
CELLS = SOURCE["cells"]


def text(i):
    return "".join(CELLS[i]["source"])


def code(src):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src.splitlines(keepends=True))


def md(src):
    return dict(cell_type="markdown", metadata={}, source=src.splitlines(keepends=True))


def replace_once(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f"Expected exactly one {old!r} in the stage-2 notebook; it changed, update this generator")
    return src.replace(old, new)


def bootstrap(extra_files):
    """Stage-2 cell 2 (clone, credentials, GPU check) with the start time recorded and extra files required."""
    src = text(2)
    src = replace_once(src, "import os, sys, subprocess, re\n", "import os, sys, subprocess, re, time\nNOTEBOOK_START = time.time()\n")
    files = '"configs/stage2.json"):'
    src = replace_once(src, files, '"configs/stage2.json", ' + ", ".join(f'"{f}"' for f in extra_files) + "):")
    return src


RUN_LOGGED = '''
import shutil, time


def tail(path, lines=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            return "\\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_logged(command, log, tail_every=300):
    """Run a command with its output in a FILE (a stalled Kaggle log stream can block a chatty process)."""
    os.environ["PYTHONFAULTHANDLER"] = "1"
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Output ->", log, flush=True)
    with log.open("ab") as console:
        process = subprocess.Popen(command, cwd=REPO, stdout=console, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        last = time.time()
        while process.poll() is None:
            time.sleep(15)
            if time.time() - last >= tail_every:
                print(f"--- {time.strftime('%H:%M:%S')} last lines ---", flush=True)
                print(tail(log, 6), flush=True)
                last = time.time()
    print(tail(log, 60), flush=True)
    if process.returncode:
        raise RuntimeError(f"{Path(command[2]).name} exited with {process.returncode}; the traceback is above")
'''

DATA_INTRO = """# Urdu Orpheus stage 3: build the v4 dataset (UAT + ASLP + UrduTTS)

**One run, no decisions.** Before you start: enable **GPU (T4)** and **Internet**, attach the `HF_TOKEN` secret
(Add-ons > Secrets, a Hugging Face token with write access to the owner's checkpoint repo), then
**Save Version > Save & Run All**. Takes about 2.5-3 hours of GPU. Nothing else is needed.

What it does:
1. A smoke test of the whole pipeline on 48 clips (about 10 minutes). If this fails, nothing was published.
2. Downloads the UAT corpus, converts it to 24 kHz mono, measures audio quality (DNSMOS/VAD).
3. Checks every UAT transcript against its audio with MMS CTC alignment and drops the worst-matching clips.
4. Re-selects the ASLP conversational and UrduTTS clips, builds a ~40 h train / validation split with voice labels.
5. Verifies every file, uploads the private dataset, and publishes the pointer the training notebook reads.
6. Loads the published dataset with the trainer's own loader as a final consistency check.

It succeeded if the last cell prints `STAGE-3 DATA READY`. Then run the training notebook.
UAT has no declared licence: the dataset it creates is private and for this course project only.
"""

DATA_SMOKE = '''
# 1/2 Smoke test: the whole pipeline on 48 UAT clips and one part of each old pool. Nothing is uploaded.
SMOKE = Path("/kaggle/working/stage3-smoke")
run_logged([TRAIN_PYTHON, "-u", str(REPO / "stage3_build.py"), "--phase", "all", "--work", str(SMOKE),
            "--limit", "48", "--pool-parts", "1", "--no-hub", "--allow-out-of-range"], "/kaggle/working/stage3_smoke.log")
print("SMOKE TEST PASSED: ingest, scoring, alignment, selection and verification all worked.")
shutil.rmtree(SMOKE, ignore_errors=True)
'''

DATA_FULL = '''
# 2/2 The real build. Resumable: if the session dies, run the notebook again and it continues.
if SMOKE_ONLY:
    print("SMOKE_ONLY is set: stopping before the full build.")
else:
  run_logged([TRAIN_PYTHON, "-u", str(REPO / "stage3_build.py"), "--phase", "all", "--work", "/kaggle/working/stage3"],
             "/kaggle/working/stage3_full.log", tail_every=600)
'''

DATA_CHECK = '''
def consumer_check():
    # Load the published dataset with the trainer's own code (first shard only) and print the summary.
    import json
    from huggingface_hub import hf_hub_download
    pointer = json.loads(Path(hf_hub_download(HF_REPO_ID, "datasets/stage3-v4.json")).read_text())
    print("Pointer:", pointer)
    report = json.loads(Path(hf_hub_download(pointer["dataset_id"], "report.json", repo_type="dataset",
                                             revision=pointer["dataset_revision"])).read_text())
    print("Train:", json.dumps(report["train"], indent=1, ensure_ascii=False))
    print("UAT screening:", json.dumps(report["uat"], indent=1, ensure_ascii=False))
    print("Conversational voice candidates:", json.dumps(report["conversational_voice_candidates"], ensure_ascii=False))
    check = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
import stage2_plan, stage2_data
base = stage2_plan.load_base({str(REPO)!r})
cfg = stage2_plan.epoch_config(6, base, data_pointer=stage2_plan.load_data_pointer())
cfg.update(test_train_clips=24, test_validation_clips=8, work_dir="/kaggle/working/prep-check")
print("Loaded by the trainer:", stage2_data.prepare_dataset(cfg, cfg["work_dir"]))
"""
    subprocess.run([TRAIN_PYTHON, "-c", check], cwd=REPO, check=True)
    shutil.rmtree("/kaggle/working/prep-check", ignore_errors=True)
    print("STAGE-3 DATA READY")


if SMOKE_ONLY:
    print("SMOKE_ONLY: the smoke test passed and nothing was published.")
else:
    consumer_check()
'''

TRAIN_INTRO = """# Urdu Orpheus stage 3: epoch 6 on the v4 dataset, smoke test then full run in one go

**One run, no decisions.** Before you start: enable **GPU (T4)** and **Internet**, attach the `HF_TOKEN` secret
(Add-ons > Secrets, a Hugging Face token with write access to the owner's checkpoint repo), then
**Save Version > Save & Run All**. Takes about 10 hours; the notebook stops and saves before Kaggle's 12-hour limit.

What it does, in order:
1. Installs the environment and runs the unit tests.
2. A short smoke run (24 clips: training, checkpoint, Hub backup, final adapter). If it fails, the full run never starts.
3. The full epoch 6 from the finished epoch-5 adapter on the dataset the data notebook published.
   Progress, checkpoints and logs are on the Hub, so if Kaggle stops the session, **just run this notebook again**:
   it resumes from the last verified checkpoint (same commit, same epoch).

It succeeded if the status cell at the bottom shows `complete`. Do not change anything in the cells.
"""

TRAIN_LOOP_HEAD = '''
# Smoke run first, then the full epoch, in one notebook run. A failure in either stops everything.
from huggingface_hub import HfApi
POINTER = None
sys.path.insert(0, str(REPO))
if EPOCH >= 6:
    import stage2_plan as _plan
    POINTER = _plan.load_data_pointer(os.environ["HF_TOKEN"])
    print("Dataset pointer:", POINTER)
STAGES = [True, False] if RUN_MODE == "smoke_then_full" else [RUN_MODE == "smoke"]
for TEST in STAGES:
'''


def train_stage_body():
    config = text(6)
    config = replace_once(config, "cfg = stage2_plan.epoch_config(EPOCH, base, LEARNING_RATE)",
                          "cfg = stage2_plan.epoch_config(EPOCH, base, LEARNING_RATE, data_pointer=POINTER)")
    config = replace_once(config, "session_hours=SESSION_HOURS)",
                          "session_hours=max(1.0, SESSION_HOURS - (time.time() - NOTEBOOK_START) / 3600))")
    # A matching smoke receipt for this commit/settings already exists (e.g. a restarted notebook): skip the smoke run.
    skip = '''
if TEST:
    from workflow_gate import recipe_key
    _name = "workflow/smoke-" + recipe_key(cfg, PROJECT_COMMIT) + ".json"
    if HfApi().file_exists(HF_REPO_ID, _name):
        print("A passing smoke receipt already exists for this commit and settings; skipping the smoke run.")
        continue
'''
    config = replace_once(config, 'RUN = Path(cfg["work_dir"]) / "runs" / cfg["run_id"]\n',
                          'RUN = Path(cfg["work_dir"]) / "runs" / cfg["run_id"]\n' + skip)
    run = replace_once(text(9), 'print("Set TEST=False (EPOCH=2) and run again for the full second epoch.")',
                       'print("Smoke test passed; the full epoch starts next.")')
    body = config + "\n" + text(7) + "\n" + run
    return textwrap.indent(body, "    ")


def build_train():
    cells = [md(TRAIN_INTRO), code(bootstrap(["stage3_build.py"]).replace(
        'EPOCH = 2  # 2 = from UrduTTS epoch 1; 3, 4, ... = from the finished previous stage-2 epoch.',
        'EPOCH = 6  # Stage 3 trains epoch 6 (then 7, 8) on the v4 dataset; each starts from the previous epoch.\n'
        'RUN_MODE = "smoke_then_full"  # "smoke_then_full" (default), "smoke" or "full" (needs an existing smoke receipt)'))]
    cells += [md("Cell 2: install and preflight"), code(text(4))]
    cells += [md("Cell 3: smoke run, then the full epoch"), code(TRAIN_LOOP_HEAD + train_stage_body())]
    cells += [md("Cell 4: status"), code(text(11))]
    return cells


def build_data():
    boot = bootstrap(["stage3_build.py", "build_stage2.py", "stage2_labels.py"])
    boot = replace_once(boot, "EPOCH = 2  # 2 = from UrduTTS epoch 1; 3, 4, ... = from the finished previous stage-2 epoch.",
                        'SMOKE_ONLY = False  # True: run only the 10-minute smoke test (debugging); the full build is skipped.')
    cells = [md(DATA_INTRO), code(boot), md("Cell 2: install and preflight"), code(text(4)),
             md("Cell 3: smoke test of the pipeline"), code(RUN_LOGGED + DATA_SMOKE),
             md("Cell 4: build and publish the dataset"), code(DATA_FULL),
             md("Cell 5: consumer check"), code(DATA_CHECK)]
    return cells


def write(name, cells):
    nb = dict(SOURCE, cells=cells)
    (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    compile("\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code"), name, "exec")  # syntax check
    print("wrote", name, f"({len(cells)} cells)")


if __name__ == "__main__":
    write("kaggle_stage3_data.ipynb", build_data())
    write("kaggle_stage3_train.ipynb", build_train())
