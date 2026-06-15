"""P-value definitions for circular-shift permutation analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircularShiftPValues:
    observed_beta: float
    perm_beta_mean: float
    perm_beta_median: float
    perm_beta_sd: float
    n_perm: int
    n_extreme_median_centered_two_sided: int
    empirical_p_median_centered_two_sided: float
    empirical_p_median_centered_two_sided_plus1: float
    n_extreme_lower_tail: int
    empirical_p_lower_tail: float
    empirical_p_lower_tail_plus1: float
    n_extreme_upper_tail: int
    empirical_p_upper_tail: float
    empirical_p_doubled_smaller_tail: float
    empirical_p_doubled_smaller_tail_plus1: float
    n_extreme_abs_coefficient: int
    empirical_p_abs_coefficient: float
    empirical_p_abs_coefficient_plus1: float
    median_centered_distance_observed: float
    median_centered_lower_cutoff: float
    median_centered_upper_cutoff: float
    min_possible_positive_p: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "observed_beta": self.observed_beta,
            "perm_beta_mean": self.perm_beta_mean,
            "perm_beta_median": self.perm_beta_median,
            "perm_beta_sd": self.perm_beta_sd,
            "n_perm": self.n_perm,
            "n_extreme_median_centered_two_sided": self.n_extreme_median_centered_two_sided,
            "empirical_p_median_centered_two_sided": self.empirical_p_median_centered_two_sided,
            "empirical_p_median_centered_two_sided_plus1": self.empirical_p_median_centered_two_sided_plus1,
            "n_extreme_lower_tail": self.n_extreme_lower_tail,
            "empirical_p_lower_tail": self.empirical_p_lower_tail,
            "empirical_p_lower_tail_plus1": self.empirical_p_lower_tail_plus1,
            "n_extreme_upper_tail": self.n_extreme_upper_tail,
            "empirical_p_upper_tail": self.empirical_p_upper_tail,
            "empirical_p_doubled_smaller_tail": self.empirical_p_doubled_smaller_tail,
            "empirical_p_doubled_smaller_tail_plus1": self.empirical_p_doubled_smaller_tail_plus1,
            "n_extreme_abs_coefficient": self.n_extreme_abs_coefficient,
            "empirical_p_abs_coefficient": self.empirical_p_abs_coefficient,
            "empirical_p_abs_coefficient_plus1": self.empirical_p_abs_coefficient_plus1,
            "median_centered_distance_observed": self.median_centered_distance_observed,
            "median_centered_lower_cutoff": self.median_centered_lower_cutoff,
            "median_centered_upper_cutoff": self.median_centered_upper_cutoff,
            "min_possible_positive_p": self.min_possible_positive_p,
        }


def _empirical_p(n_extreme: int, n_perm: int) -> float:
    return float(n_extreme / n_perm)


def _plus_one_p(n_extreme: int, n_perm: int) -> float:
    return float((n_extreme + 1) / (n_perm + 1))


def circular_shift_pvalues(
    perm_betas: np.ndarray,
    observed_beta: float,
) -> CircularShiftPValues:
    """Calculate circular-shift empirical P-values from saved permutation betas.

    The primary two-sided definition is median-centered:
    |beta_perm - median(beta_perm)| >= |beta_observed - median(beta_perm)|.
    The lower-tail and old absolute-coefficient definitions are retained as
    explicitly named secondary quantities.
    """

    perm = np.asarray(perm_betas, dtype=float)
    if perm.ndim != 1 or len(perm) == 0:
        raise ValueError("perm_betas must be a non-empty 1D array.")
    if not np.isfinite(observed_beta):
        raise ValueError("observed_beta must be finite.")
    if not np.all(np.isfinite(perm)):
        raise ValueError("perm_betas contains non-finite values.")

    n_perm = int(len(perm))
    observed = float(observed_beta)
    median = float(np.median(perm))
    distance_observed = float(abs(observed - median))
    lower_cutoff = float(median - distance_observed)
    upper_cutoff = float(median + distance_observed)

    n_median_centered = int(np.sum(np.abs(perm - median) >= distance_observed))
    n_lower = int(np.sum(perm <= observed))
    n_upper = int(np.sum(perm >= observed))
    n_abs = int(np.sum(np.abs(perm) >= abs(observed)))

    p_lower_plus1 = _plus_one_p(n_lower, n_perm)
    p_upper_plus1 = _plus_one_p(n_upper, n_perm)

    return CircularShiftPValues(
        observed_beta=observed,
        perm_beta_mean=float(np.mean(perm)),
        perm_beta_median=median,
        perm_beta_sd=float(np.std(perm, ddof=0)),
        n_perm=n_perm,
        n_extreme_median_centered_two_sided=n_median_centered,
        empirical_p_median_centered_two_sided=_empirical_p(n_median_centered, n_perm),
        empirical_p_median_centered_two_sided_plus1=_plus_one_p(n_median_centered, n_perm),
        n_extreme_lower_tail=n_lower,
        empirical_p_lower_tail=_empirical_p(n_lower, n_perm),
        empirical_p_lower_tail_plus1=p_lower_plus1,
        n_extreme_upper_tail=n_upper,
        empirical_p_upper_tail=_empirical_p(n_upper, n_perm),
        empirical_p_doubled_smaller_tail=min(
            1.0, 2.0 * min(_empirical_p(n_lower, n_perm), _empirical_p(n_upper, n_perm))
        ),
        empirical_p_doubled_smaller_tail_plus1=min(
            1.0, 2.0 * min(p_lower_plus1, p_upper_plus1)
        ),
        n_extreme_abs_coefficient=n_abs,
        empirical_p_abs_coefficient=_empirical_p(n_abs, n_perm),
        empirical_p_abs_coefficient_plus1=_plus_one_p(n_abs, n_perm),
        median_centered_distance_observed=distance_observed,
        median_centered_lower_cutoff=lower_cutoff,
        median_centered_upper_cutoff=upper_cutoff,
        min_possible_positive_p=float(1.0 / n_perm),
    )
