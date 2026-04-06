"""Statistical analysis engine implemented with pure NumPy.

scipy is not used because the installed scipy 1.8.0 has a broken liblapack
dependency on this system. All required statistics (Mann-Whitney U, t-
distribution critical values, quantiles) are implemented with NumPy.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from schemas.profile_schema import ProfileRecord


@dataclass
class SignificanceResult:
    statistic: float
    p_value: float
    significant: bool


def _records_to_df(records: List[ProfileRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


# ---------------------------------------------------------------------------
# Pure-NumPy Mann-Whitney U implementation
# ---------------------------------------------------------------------------

def _mannwhitneyu(x: np.ndarray, y: np.ndarray) -> tuple:
    """Two-sided Mann-Whitney U test using the normal approximation.

    Returns (U_statistic, p_value).
    """
    n1, n2 = len(x), len(y)
    # Rank the combined sample
    combined = np.concatenate([x, y])
    order = combined.argsort()
    ranks = np.empty(len(combined))
    # Handle ties by averaging
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and combined[order[j]] == combined[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_x = ranks[:n1].sum()
    U1 = rank_sum_x - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Normal approximation (with tie correction)
    mu_U = n1 * n2 / 2.0
    # Tie correction
    combined_sorted = np.sort(combined)
    _, counts = np.unique(combined_sorted, return_counts=True)
    tie_correction = np.sum(counts ** 3 - counts) / 12.0
    N = n1 + n2
    sigma_U = np.sqrt(
        n1 * n2 / (N * (N - 1)) * (N ** 3 - N) / 12.0 - n1 * n2 * tie_correction / (N * (N - 1))
    )

    if sigma_U == 0:
        # All values are identical
        return float(U), 1.0

    z = (U - mu_U) / sigma_U
    # Two-sided p-value via standard normal CDF approximation
    p_value = 2.0 * _norm_sf(abs(z))
    return float(U), float(np.clip(p_value, 0.0, 1.0))


def _norm_sf(z: float) -> float:
    """Survival function of the standard normal distribution (P(Z > z)).

    Uses the complementary error function available in numpy/math.
    """
    return float(0.5 * np.exp(-0.5 * z * z) * _erfc_approx(z / np.sqrt(2.0)) / 0.5)


def _norm_cdf(z: float) -> float:
    """CDF of the standard normal using the error function."""
    return 0.5 * (1.0 + _erf_approx(z / np.sqrt(2.0)))


def _norm_sf(z: float) -> float:  # noqa: F811
    return 1.0 - _norm_cdf(z)


def _erf_approx(x: float) -> float:
    """Compute erf(x) via numpy's built-in erfinv-free path."""
    # numpy doesn't expose erf directly but math.erf is always available
    import math
    return math.erf(x)


def _erfc_approx(x: float) -> float:
    import math
    return math.erfc(x)


# ---------------------------------------------------------------------------
# t-distribution quantile (percent point function) via numerical inversion
# ---------------------------------------------------------------------------

def _t_ppf(p: float, df: int) -> float:
    """Percent-point function of Student's t-distribution.

    Uses a rational approximation for the standard normal (Abramowitz & Stegun
    26.2.17) and the relationship between normal and t quantiles via the
    Cornish-Fisher expansion for moderate df, falling back to a simple
    bisection search for accuracy.
    """
    if df >= 30:
        # For large df, t ≈ normal
        return _norm_ppf(p)
    # Bisection search on the t CDF
    lo, hi = -100.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _t_cdf(t: float, df: int) -> float:
    """CDF of Student's t-distribution via the regularised incomplete beta function."""
    import math
    x = df / (df + t * t)
    ib = _regularised_incomplete_beta(df / 2.0, 0.5, x)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute I_x(a, b) using a continued-fraction expansion (Lentz method)."""
    import math
    if x < 0 or x > 1:
        raise ValueError("x must be in [0, 1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    # Continued fraction via Lentz
    cf = _betacf(a, b, x)
    return front * cf


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularised incomplete beta (Numerical Recipes)."""
    MAXIT = 200
    EPS = 3e-12
    FPMIN = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (rational approximation, Abramowitz & Stegun 26.2.17)."""
    import math
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    if p < 0.5:
        sign = -1.0
        q = p
    else:
        sign = 1.0
        q = 1.0 - p
    t = math.sqrt(-2.0 * math.log(q))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return sign * (t - num / den)


# ---------------------------------------------------------------------------
# StatisticsEngine
# ---------------------------------------------------------------------------

class StatisticsEngine:
    """Compute descriptive statistics and hypothesis tests over ProfileRecords."""

    def summarize(
        self,
        records: List[ProfileRecord],
        group_by: List[str],
        metric: str,
    ) -> pd.DataFrame:
        """Aggregate ``metric`` per ``group_by`` group.

        Computed columns per group:
        mean, std, median, p95, p99, ci_95_lower, ci_95_upper, count.
        """
        df = _records_to_df(records)

        def agg(series: pd.Series) -> pd.Series:
            vals = series.to_numpy(dtype=float)
            n = len(vals)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0

            if n > 1:
                se = std / np.sqrt(n)
                t_crit = _t_ppf(0.975, df=n - 1)
                ci_lower = mean - t_crit * se
                ci_upper = mean + t_crit * se
            else:
                ci_lower = mean
                ci_upper = mean

            return pd.Series(
                {
                    "mean": mean,
                    "std": std,
                    "median": float(np.median(vals)),
                    "p95": float(np.percentile(vals, 95)),
                    "p99": float(np.percentile(vals, 99)),
                    "ci_95_lower": ci_lower,
                    "ci_95_upper": ci_upper,
                    "count": float(n),
                }
            )

        result = df.groupby(group_by)[metric].apply(agg).unstack(level=-1).reset_index()
        return result

    def test_significance(
        self,
        group_a: List[ProfileRecord],
        group_b: List[ProfileRecord],
        metric: str,
        alpha: float = 0.05,
    ) -> SignificanceResult:
        """Perform a two-sided Mann-Whitney U test comparing ``metric`` between groups."""
        vals_a = np.array([getattr(r, metric) for r in group_a], dtype=float)
        vals_b = np.array([getattr(r, metric) for r in group_b], dtype=float)

        u_stat, p_value = _mannwhitneyu(vals_a, vals_b)
        return SignificanceResult(
            statistic=u_stat,
            p_value=p_value,
            significant=bool(p_value < alpha),
        )

    def compute_overhead(
        self,
        with_callback: List[ProfileRecord],
        without_callback: List[ProfileRecord],
        metric: str,
    ) -> float:
        """Compute percentage overhead: (mean_with - mean_without) / mean_without * 100."""
        mean_with = float(np.mean([getattr(r, metric) for r in with_callback]))
        mean_without = float(np.mean([getattr(r, metric) for r in without_callback]))

        if mean_without == 0:
            return 0.0

        return (mean_with - mean_without) / mean_without * 100.0
