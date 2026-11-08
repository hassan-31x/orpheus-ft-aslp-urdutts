"""Generate the same unseen prompts/seeds for UrduTTS and mixed-data adapters."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="Staged init_adapter directory")
    parser.add_argument("--mixed", type=Path, required=True, help="Final mixed adapter directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    for label, adapter in (("urdutts_parent", args.parent), ("mixed", args.mixed)):
        if not (adapter/"adapter_model.safetensors").is_file() or not (adapter/"run_identity.json").is_file():
            raise RuntimeError(f"Missing adapter weights/identity for {label}")
        subprocess.run([sys.executable, str(repo/"synthesize.py"), "--adapter", str(adapter),
                        "--manifest", str(repo/"benchmark_prompts.csv"), "--output", str(args.output/label),
                        "--seed", str(args.seed)], check=True)
    (args.output/"comparison.json").write_text(json.dumps(dict(parent=str(args.parent), mixed=str(args.mixed),
        seed=args.seed, protocol="Identical text/decoding/seeds. Blind listening required; no automatic naturalness claim."), indent=2))


if __name__ == "__main__":
    main()
