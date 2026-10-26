from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class PairedMetricStats:
    n: int
    target_mean: float
    target_std: float
    baseline_mean: float
    baseline_std: float
    mean_diff: float
    mean_diff_ci_low: float
    mean_diff_ci_high: float
    wilcoxon_stat: float
    wilcoxon_p: float
    cliffs_delta: float
    cliffs_magnitude: str


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Cliff's delta for two paired-length score lists."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_x = len(x_arr)
    n_y = len(y_arr)
    if n_x == 0 or n_y == 0:
        raise ValueError("Cliff's delta requires non-empty samples.")

    gt = 0
    lt = 0
    for value_x in x_arr:
        diffs = value_x - y_arr
        gt += int(np.sum(diffs > 0))
        lt += int(np.sum(diffs < 0))
    return (gt - lt) / float(n_x * n_y)


def cliffs_delta_magnitude(delta: float) -> str:
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.33:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def paired_bootstrap_mean_diff_ci(
    target: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 2024,
) -> Tuple[float, float]:
    """Bootstrap the paired mean difference CI by resampling aligned subject pairs."""
    target_arr = np.asarray(target, dtype=float)
    baseline_arr = np.asarray(baseline, dtype=float)
    if target_arr.shape != baseline_arr.shape:
        raise ValueError("Paired bootstrap requires aligned arrays of the same shape.")
    if len(target_arr) == 0:
        raise ValueError("Paired bootstrap requires at least one paired observation.")

    diffs = target_arr - baseline_arr
    rng = np.random.default_rng(random_seed)
    samples = np.empty(n_resamples, dtype=float)
    n = len(diffs)
    for idx in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        samples[idx] = float(np.mean(diffs[sample_idx]))

    alpha = 1.0 - confidence_level
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return low, high


def safe_wilcoxon(target: Sequence[float], baseline: Sequence[float]) -> Tuple[float, float]:
    """Run two-sided Wilcoxon while handling identical paired vectors."""
    target_arr = np.asarray(target, dtype=float)
    baseline_arr = np.asarray(baseline, dtype=float)
    diffs = target_arr - baseline_arr
    if np.allclose(diffs, 0.0):
        return 0.0, 1.0
    stat, p_value = wilcoxon(target_arr, baseline_arr, alternative="two-sided")
    return float(stat), float(p_value)


def compute_paired_metric_stats(
    target: Sequence[float],
    baseline: Sequence[float],
    *,
    bootstrap_resamples: int = 1000,
    random_seed: int = 2024,
) -> PairedMetricStats:
    target_arr = np.asarray(target, dtype=float)
    baseline_arr = np.asarray(baseline, dtype=float)
    if target_arr.shape != baseline_arr.shape:
        raise ValueError("Paired statistics require aligned arrays of the same shape.")
    if len(target_arr) == 0:
        raise ValueError("Paired statistics require at least one paired observation.")

    wilcoxon_stat, wilcoxon_p = safe_wilcoxon(target_arr, baseline_arr)
    delta = cliffs_delta(target_arr, baseline_arr)
    ci_low, ci_high = paired_bootstrap_mean_diff_ci(
        target_arr,
        baseline_arr,
        n_resamples=bootstrap_resamples,
        random_seed=random_seed,
    )

    return PairedMetricStats(
        n=len(target_arr),
        target_mean=float(np.mean(target_arr)),
        target_std=float(np.std(target_arr, ddof=1)),
        baseline_mean=float(np.mean(baseline_arr)),
        baseline_std=float(np.std(baseline_arr, ddof=1)),
        mean_diff=float(np.mean(target_arr - baseline_arr)),
        mean_diff_ci_low=ci_low,
        mean_diff_ci_high=ci_high,
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_p=wilcoxon_p,
        cliffs_delta=delta,
        cliffs_magnitude=cliffs_delta_magnitude(delta),
    )


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values."""
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return []

    order = np.argsort(p_arr)
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        factor = m - rank
        adjusted = factor * p_arr[idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(1.0, running_max)

    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def significance_marker(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def apply_holm_by_groups(rows: List[Dict[str, object]], group_keys: Iterable[str]) -> List[Dict[str, object]]:
    """Apply Holm adjustment to rows that already contain a raw `wilcoxon_p` field."""
    group_keys = list(group_keys)
    grouped: Dict[Tuple[object, ...], List[int]] = {}
    for idx, row in enumerate(rows):
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(idx)

    for indices in grouped.values():
        adjusted = holm_bonferroni_adjust([float(rows[idx]["wilcoxon_p"]) for idx in indices])
        for idx, p_adj in zip(indices, adjusted):
            rows[idx]["wilcoxon_p_adj"] = p_adj
            rows[idx]["significance_raw"] = significance_marker(float(rows[idx]["wilcoxon_p"]))
            rows[idx]["significance_adj"] = significance_marker(p_adj)
    return rows
