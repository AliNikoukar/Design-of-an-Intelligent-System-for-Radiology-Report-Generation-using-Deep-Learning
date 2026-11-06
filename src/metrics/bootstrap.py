"""
Statistical significance and bootstrap confidence intervals.
Computes non-parametric 95% confidence intervals via paired resampling.
"""

from typing import Dict, Tuple, List, Callable, Any
import numpy as np


def compute_bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable[[np.ndarray], float] = np.mean,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes non-parametric bootstrap confidence intervals.

    Args:
        data: Array of sample scores or values.
        stat_func: Statistic to compute (e.g. np.mean).
        n_bootstraps: Number of resampling iterations.
        confidence_level: e.g. 0.95 for 95% CI.
        seed: Random seed for reproducibility.

    Returns:
        (mean_val, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = []

    for _ in range(n_bootstraps):
        sample_indices = rng.randint(0, n, size=n)
        sample = data[sample_indices]
        boot_stats.append(stat_func(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_stats, 100.0 * alpha))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    mean_val = float(np.mean(boot_stats))

    return mean_val, lower, upper
