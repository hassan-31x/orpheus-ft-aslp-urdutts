"""Screen in an isolated process so ASR releases GPU memory before SNAC/Orpheus."""
import argparse
import logging
from pathlib import Path

from orpheus_utils import atomic_json, read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = read_json(args.config)
    if cfg.get("stage2_dataset"):
        from stage2_data import prepare_dataset
    else:
        from mixed_data import prepare_dataset
    root = prepare_dataset(cfg, Path(cfg["work_dir"]))
    atomic_json(args.result, dict(root=str(root)))
