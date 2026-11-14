"""Publish and run the stage-2 notebook on the authenticated Kaggle account (any account works).

  python kaggle_submit_stage2.py smoke                 # epoch-2 few-clip test
  python kaggle_submit_stage2.py full --epoch 2        # full epoch 2 (or resume it on another account)
  python kaggle_submit_stage2.py full --epoch 3        # next epochs: same command, new --epoch
  python kaggle_submit_stage2.py eval --eval-name NAME --adapter label=repo:run_id:checkpoint ...
  python kaggle_submit_stage2.py listen --adapter "display label=repo:run_id:checkpoint" ...   # audio players in the notebook

The commit pinned inside the notebook is --ref (default: this checkout's HEAD, which must be pushed).
Keep --ref identical for all epochs and accounts. The account needs the HF_TOKEN Kaggle secret enabled
for the notebook it runs (Kaggle only allows granting that in the web UI for a NEW notebook; reusing an
existing notebook slug that already has the secret, via --kernel, avoids it).
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["smoke", "full", "eval", "listen"])
    p.add_argument("--adapter", action="append", default=[], help="eval/listen: label=repo:run_id:checkpoint (repeatable)")
    p.add_argument("--eval-name", help="eval: results folder under evaluations/")
    p.add_argument("--epoch", type=int, default=2)
    p.add_argument("--ref", help="40-character commit pushed to GitHub (default: local HEAD)")
    p.add_argument("--session-hours", type=float, default=11.75, help="Stop-and-save budget: Kaggle's 12 h session limit minus a 15 min save margin")
    p.add_argument("--kernel", help="owner/slug to publish to (default: <you>/orpheus-urdu-stage2-<action>)")
    p.add_argument("--token-dataset", help="Private Kaggle dataset (owner/slug) containing hf_token.txt; used when the HF_TOKEN secret "
                   "cannot be read in API-started sessions")
    p.add_argument("--no-run", action="store_true")
    args = p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value(api.CONFIG_NAME_USER)
    repo = Path(__file__).resolve().parent
    ref = args.ref or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", ref], cwd=repo, capture_output=True, text=True).stdout
    if not pushed.strip():
        raise SystemExit(f"Commit {ref} is not on a pushed remote branch; push it first (Kaggle clones from GitHub)")
    source = {"eval": "kaggle_eval_pron.ipynb", "listen": "kaggle_listen.ipynb"}.get(args.action, "kaggle_stage2.ipynb")
    notebook = json.loads((repo / source).read_text())
    replacements = {"REPO_REF = ": f'REPO_REF = "{ref}"', "EPOCH = ": f"EPOCH = {args.epoch}",
                    "TEST = ": f"TEST = {args.action == 'smoke'}", "SESSION_HOURS = ": f"SESSION_HOURS = {args.session_hours}"}
    if args.action in ("eval", "listen"):
        if not args.adapter or (args.action == "eval" and not args.eval_name):
            raise SystemExit("eval needs --eval-name and at least one --adapter; listen needs at least one --adapter")
        replacements["ADAPTERS = "] = f"ADAPTERS = {json.dumps(args.adapter, ensure_ascii=False)}"
        if args.action == "eval":
            replacements["EVAL_NAME = "] = f"EVAL_NAME = {json.dumps(args.eval_name)}"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines(keepends=True)
        for i, line in enumerate(lines):
            for prefix, new in replacements.items():
                if line.startswith(prefix):
                    comment = line.split("  #", 1)[1].rstrip("\n") if "  #" in line else ""
                    lines[i] = new + ("  #" + comment if comment else "") + "\n"
        cell["source"] = lines
    kernel = args.kernel or f"{user}/orpheus-urdu-stage2-{args.action}"
    if not kernel.startswith(user + "/"):
        raise SystemExit("Use a notebook owned by the authenticated Kaggle account")
    meta = dict(id=kernel, title=kernel.split("/")[1].replace("-", " "), code_file="kaggle_stage2.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=True, enable_tpu=False,
                dataset_sources=[args.token_dataset] if args.token_dataset else [], competition_sources=[], kernel_sources=[], model_sources=[])
    with tempfile.TemporaryDirectory(prefix="stage2-kaggle-") as d:
        (Path(d) / "kaggle_stage2.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1))  # code_file name
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        cmd = ["kaggle", "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4"] + (["--no-run"] if args.no_run else [])
        subprocess.run(cmd, check=True)
    print(f"Notebook: https://www.kaggle.com/code/{kernel} | action={args.action} epoch={args.epoch} ref={ref[:10]}")


if __name__ == "__main__":
    main()
