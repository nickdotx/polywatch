"""Tests for the Phase-2 statistics primitives (DSR/PSR/t-stat), checked against
known values. No network, stdlib only."""

import math

from polywatch import stats


# --- normal distribution helpers ---------------------------------------

def test_norm_cdf_known_points():
    assert abs(stats.norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(stats.norm_cdf(1.959963985) - 0.975) < 1e-6
    assert abs(stats.norm_cdf(-1.959963985) - 0.025) < 1e-6


def test_norm_ppf_known_points():
    assert abs(stats.norm_ppf(0.5)) < 1e-9
    assert abs(stats.norm_ppf(0.975) - 1.959963985) < 1e-6
    assert abs(stats.norm_ppf(0.99) - 2.326347874) < 1e-6


def test_ppf_cdf_roundtrip():
    for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        assert abs(stats.norm_cdf(stats.norm_ppf(p)) - p) < 1e-9


# --- moments ------------------------------------------------------------

def test_mean_and_stdev():
    xs = [1, 2, 3, 4, 5]
    assert abs(stats.mean(xs) - 3.0) < 1e-12
    # sample stdev (ddof=1) of 1..5 is sqrt(2.5)
    assert abs(stats.stdev(xs) - math.sqrt(2.5)) < 1e-12


def test_skewness_symmetric_is_zero():
    assert abs(stats.skewness([-2, -1, 0, 1, 2])) < 1e-9


def test_kurtosis_normal_is_about_three():
    # uniform-ish symmetric set: just assert it returns the non-excess scale
    # (normal = 3). A flat/platykurtic set should be < 3.
    assert stats.kurtosis([-2, -1, 0, 1, 2]) < 3.0


# --- sharpe / t-stat ----------------------------------------------------

def test_tstat_equals_sharpe_times_sqrt_n():
    xs = [0.2, -0.1, 0.3, 0.05, -0.05, 0.15, 0.1, -0.2]
    n = len(xs)
    assert abs(stats.t_stat(xs) - stats.sharpe_ratio(xs) * math.sqrt(n)) < 1e-9


def test_tstat_zero_when_no_variance():
    assert stats.t_stat([0.1, 0.1, 0.1]) == 0.0


# --- PSR ----------------------------------------------------------------

def test_psr_half_when_sr_equals_benchmark():
    # z=0 -> Phi(0)=0.5 regardless of n
    psr = stats.probabilistic_sharpe_ratio(0.3, n=100, skew=0.0, kurt=3.0,
                                           sr_benchmark=0.3)
    assert abs(psr - 0.5) < 1e-9


def test_psr_monotonic_in_track_length():
    # Same observed Sharpe above benchmark: more observations -> more confidence
    a = stats.probabilistic_sharpe_ratio(0.2, 30, 0.0, 3.0, 0.0)
    b = stats.probabilistic_sharpe_ratio(0.2, 300, 0.0, 3.0, 0.0)
    assert b > a > 0.5


def test_psr_penalizes_fat_tails_and_negative_skew():
    base = stats.probabilistic_sharpe_ratio(0.2, 100, 0.0, 3.0, 0.0)
    fat = stats.probabilistic_sharpe_ratio(0.2, 100, 0.0, 9.0, 0.0)
    neg_skew = stats.probabilistic_sharpe_ratio(0.2, 100, -1.5, 3.0, 0.0)
    assert fat < base          # fatter tails -> less confident
    assert neg_skew < base     # negative skew -> less confident


# --- expected max sharpe / DSR -----------------------------------------

def test_expected_max_sharpe_grows_with_trials():
    s10 = stats.expected_max_sharpe(10, var_sr=0.04)
    s1000 = stats.expected_max_sharpe(1000, var_sr=0.04)
    assert s1000 > s10 > 0.0


def test_dsr_drops_as_more_trials_are_tested():
    # A fixed observed Sharpe looks great if you tried 5 wallets, weak if 500.
    strong = stats.deflated_sharpe_ratio(0.30, n_obs=200, n_trials=5,
                                         var_sr=0.02, skew=0.0, kurt=3.0)
    haircut = stats.deflated_sharpe_ratio(0.30, n_obs=200, n_trials=500,
                                          var_sr=0.02, skew=0.0, kurt=3.0)
    assert strong > haircut
    assert 0.0 <= haircut <= 1.0 and 0.0 <= strong <= 1.0


def test_series_stats_bundle():
    xs = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1]
    s = stats.SeriesStats.of(xs)
    assert s.n == 6
    assert abs(s.t_stat - stats.t_stat(xs)) < 1e-12
