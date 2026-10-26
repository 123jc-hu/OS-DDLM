"""Run one model on one dataset/protocol with the settings used in the paper.

Examples:
    python scripts/run.py --model DDLM-Extractor --dataset THU --train-mode single-subject
    python scripts/run.py --model "OS-Selector (Ours)" --dataset GIST --train-mode cross-subject --k 16
"""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import build_config, get_artifact_safe_name

# Dataset-specific selector schedules (Supplementary Sections S8 and S15).
SELECTOR_MODELS = {"OS-Selector (Ours)", "OS-Selector w/o Orthogonality",
                   "OS-Selector w/ Deterministic Top-K", "Gumbel-Top-K Selector"}
SELECTOR_SCHEDULES = {
    "THU": dict(selector_lr=0.01, selector_T=25, selector_beta_end=0.1),
    "GIST": dict(selector_lr=0.005, selector_T=25, selector_beta_end=0.1),
    "TCTR": dict(selector_lr=0.005, selector_T=50, selector_beta_end=0.01),
}
DDLM_ABLATIONS = ["late_downsampling", "no_dilation", "full_spatial", "no_eca"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model name as listed in configs/config.yaml.")
    parser.add_argument("--dataset", required=True,
                        choices=["THU", "GIST", "TCTR_1", "TCTR_2", "TCTR_A", "TCTR_B"])
    parser.add_argument("--train-mode", default="single-subject", choices=["single-subject", "cross-subject"])
    parser.add_argument("--k", type=int, default=None, help="Channel budget for selector models.")
    parser.add_argument("--orth", type=float, default=None, help="Override the orthogonality weight.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subjects", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None,
                        help="Inclusive participant range; defaults to all participants.")
    parser.add_argument("--ddlm-ablation", choices=DDLM_ABLATIONS, default=None,
                        help="Replace one DDLM-Extractor component (use with --model DDLM-Extractor).")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("RSVP_DATASET_ROOT"))
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    config = build_config(args.model, args.dataset)
    config["train_mode"] = args.train_mode
    if args.model in SELECTOR_MODELS:
        config.update(SELECTOR_SCHEDULES[args.dataset.split("_")[0]])
    for key, value in [("selector_K", args.k), ("selector_orth", args.orth), ("epochs", args.epochs)]:
        if value is not None:
            config[key] = value
    if args.seed is not None:
        config["random_seed"] = config["seed"] = args.seed
    if args.subjects is not None:
        config["subject_start_id"], config["subject_end_id"] = args.subjects
    if args.data_root is None:
        raise ValueError("Provide --data-root or set RSVP_DATASET_ROOT.")
    config["data_root"] = str(args.data_root)

    parts = [args.model] + ([f"K{config['selector_K']}"] if config.get("use_selector") or
                            config.get("apply_channel_selector") else [])
    if args.ddlm_ablation:
        if args.model != "DDLM-Extractor":
            raise ValueError("--ddlm-ablation applies to DDLM-Extractor only")
        from models import DDLM_Extractor
        from models.ddlm_ablation import Model
        DDLM_Extractor.Model = Model
        config["ddlm_ablation"] = args.ddlm_ablation
        parts.append(args.ddlm_ablation)
    parts.append(f"s{config['random_seed']}")
    config["model_artifact_name"] = config["log_name"] = get_artifact_safe_name("_".join(map(str, parts)))

    from main import run_experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
