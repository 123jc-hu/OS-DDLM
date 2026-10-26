"""Participant-paired comparison of one method against several others.

Each argument is a run directory that holds per-participant results, either
``results.csv`` (scripts/run.py) or ``subject_metrics.csv`` (scripts/run_resgs.py).
Pass several directories for one method, separated by commas, to average repeated
fits within each participant first. Two-sided Wilcoxon signed-rank tests are
Holm-corrected across the compared methods, separately for each metric.

Example:
    python scripts/analysis/compare_methods.py \
        --reference "experiments/OS-Selector (Ours)_K32_s2025/THU/single-subject" \
        --others experiments/SparseEA_K32_s2025/THU/single-subject experiments/ResGS/THU/single-subject/K32_s2025
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.paired_stats_core import apply_holm_by_groups, compute_paired_metric_stats

METRICS = ["BA", "AUC", "F1"]


def rank_biserial(differences):
    differences = differences[differences != 0]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    return float(np.sum(np.sign(differences) * ranks) / ranks.sum())


def load(spec):
    frames = []
    for directory in map(Path, spec.split(",")):
        if (directory / "results.csv").exists():
            frame = pd.read_csv(directory / "results.csv")
            frame = frame[frame.SUB.str.fullmatch(r"SUB\d+")].copy()  # drop the AVG/STD rows
            frame["subject"] = frame.SUB.str.extract(r"(\d+)", expand=False).astype(int)
        else:
            frame = pd.read_csv(directory / "subject_metrics.csv")
        frames.append(frame[["subject", *METRICS]])
    return pd.concat(frames).groupby("subject")[METRICS].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--others", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for the full table.")
    args = parser.parse_args()
    reference = load(args.reference)
    rows = []
    for spec in args.others:
        other = load(spec)
        common = reference.index.intersection(other.index)
        if len(common) != len(reference) or len(common) != len(other):
            raise ValueError(f"Participant sets differ: {spec}")
        for metric in METRICS:
            x, y = reference.loc[common, metric].to_numpy(), other.loc[common, metric].to_numpy()
            stats = compute_paired_metric_stats(x, y, bootstrap_resamples=10_000)
            rows.append(dict(metric=metric, other=spec, rank_biserial=rank_biserial(x - y), **asdict(stats)))
    table = pd.DataFrame(apply_holm_by_groups(rows, ["metric"]))
    columns = ["metric", "other", "n", "target_mean", "baseline_mean", "mean_diff",
               "mean_diff_ci_low", "mean_diff_ci_high", "rank_biserial", "wilcoxon_p", "wilcoxon_p_adj"]
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.output:
        table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
