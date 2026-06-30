"""Statistical primitives for honest out-of-sample validation (Phase 2).

Dependency-free (stdlib `math` only) on purpose — the project deliberately keeps
a tiny requirement set, and these are small, well-understood formulas we want to
be able to read and unit-test directly rather than trust a black box.

What lives here and why it matters for Polywatch:

- We test HUNDREDS of wallets and keep the best-looking ones. The best of many
  noisy draws is biased upward even with zero real skill. So a raw Sharpe / win
  rate / ROI on the selected set is not evidence of an edge.
- Bailey & Lopez de Prado's **Probabilistic Sharpe Ratio (PSR)** asks "what's the
  probability the true Sharpe exceeds a benchmark, given track length, skew and
  fat tails?" The **Deflated Sharpe Ratio (DSR)** is the PSR measured against a
  benchmark that is the *expected maximum* Sharpe under the null across N trials —
  i.e. it haircuts exactly for the multiple testing we do during wallet selection.
- We also expose a plain t-stat, because Harvey et al. ("factor zoo") argue for a
  t > ~3 hurdle (not 2) once multiple testing is in play.

All Sharpe values here are **per-observation** (per copy trade), not annualized;
`n` is the number of observations (copies). Track length enters through n.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EULER_MASCHERONI = 0.5772156649015329
E = math.e


# ---------------------------------------------------------------------------
# Normal distribution helpers (no scipy)
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile function).

    Acklam's rational approximation (relative error < 1.15e-9), with one
    Halley refinement step using the erf-based CDF for extra accuracy.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # One Halley step: refine using the (accurate) erf CDF.
    e_ = norm_cdf(x) - p
    u = e_ * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return x


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs, ddof: int = 1) -> float:
    """Sample standard deviation (ddof=1 by default)."""
    xs = list(xs)
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - ddof)
    return math.sqrt(var)


def skewness(xs) -> float:
    """Population skewness (gamma_3). 0 for a symmetric distribution."""
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    m2 = sum((x - m) ** 2 for x in xs) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((x - m) ** 3 for x in xs) / n
    return m3 / (m2 ** 1.5)


def kurtosis(xs) -> float:
    """Non-excess (Pearson) kurtosis (gamma_4). 3.0 for a normal distribution."""
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 3.0
    m = sum(xs) / n
    m2 = sum((x - m) ** 2 for x in xs) / n
    if m2 <= 0:
        return 3.0
    m4 = sum((x - m) ** 4 for x in xs) / n
    return m4 / (m2 ** 2)


# ---------------------------------------------------------------------------
# Sharpe / t-stat
# ---------------------------------------------------------------------------

def _degenerate(m: float, s: float) -> bool:
    """True if the dispersion is just floating-point dust around identical
    values (so a Sharpe/t-stat would explode meaninglessly)."""
    return s <= 1e-12 * (abs(m) + 1.0)


def sharpe_ratio(xs) -> float:
    """Per-observation Sharpe (mean / stdev). Not annualized."""
    xs = list(xs)
    m = mean(xs)
    s = stdev(xs)
    if s <= 0 or _degenerate(m, s):
        return 0.0
    return m / s


def t_stat(xs) -> float:
    """One-sample t-statistic for mean != 0. Equals sharpe * sqrt(n)."""
    xs = list(xs)
    n = len(xs)
    m = mean(xs)
    s = stdev(xs)
    if n < 2 or s <= 0 or _degenerate(m, s):
        return 0.0
    return (m / s) * math.sqrt(n)


def t_pvalue_one_sided(t: float, n: int) -> float:
    """Approximate one-sided p-value P(T > t) using the normal approximation.

    For n in the dozens+ (our copy counts) the normal approx is adequate; this
    is a guide, not a precise small-sample t-test.
    """
    return 1.0 - norm_cdf(t)


# ---------------------------------------------------------------------------
# Probabilistic & Deflated Sharpe Ratio (Bailey & Lopez de Prado)
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    sr_hat: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """Probability that the true (per-obs) Sharpe exceeds `sr_benchmark`.

    PSR(SR*) = Phi[ (SR_hat - SR*) * sqrt(n-1)
                    / sqrt(1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2) ]

    `skew` = gamma_3, `kurt` = gamma_4 (non-excess; 3 for normal). Track length
    enters via sqrt(n-1); fat tails / asymmetry widen the denominator.
    """
    if n < 2:
        return 0.0
    denom_var = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    if denom_var <= 0:
        denom_var = 1e-12
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_var)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, var_sr: float, mean_sr: float = 0.0) -> float:
    """Expected MAX per-obs Sharpe across `n_trials` independent strategies under
    the null, given the cross-trial variance of Sharpe estimates `var_sr`.

    E[max SR] ~= mean_sr + sqrt(var_sr) * [ (1-gamma)*Phi^-1(1 - 1/N)
                                            + gamma*Phi^-1(1 - 1/(N*e)) ]

    This is the extreme-value benchmark the DSR deflates against: even with zero
    skill, picking the best of N trials yields a positive Sharpe in expectation,
    growing with N.
    """
    if n_trials < 2 or var_sr <= 0:
        return mean_sr
    sigma = math.sqrt(var_sr)
    g = EULER_MASCHERONI
    term = (1.0 - g) * norm_ppf(1.0 - 1.0 / n_trials) + \
        g * norm_ppf(1.0 - 1.0 / (n_trials * E))
    return mean_sr + sigma * term


def deflated_sharpe_ratio(
    sr_hat: float, n_obs: int, n_trials: int, var_sr: float,
    skew: float = 0.0, kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: probability the true Sharpe is positive AFTER
    haircutting for selection across `n_trials` and for non-normal returns.

    DSR = PSR(SR0) where SR0 = expected_max_sharpe(n_trials, var_sr).
    A common pass bar is DSR > 0.95.
    """
    sr0 = expected_max_sharpe(n_trials, var_sr, mean_sr=0.0)
    return probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurt, sr_benchmark=sr0)


@dataclass
class SeriesStats:
    n: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurt: float
    t_stat: float

    @classmethod
    def of(cls, xs) -> "SeriesStats":
        xs = list(xs)
        return cls(
            n=len(xs), mean=mean(xs), std=stdev(xs), sharpe=sharpe_ratio(xs),
            skew=skewness(xs), kurt=kurtosis(xs), t_stat=t_stat(xs),
        )
