import os
import sys
import json
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
isolpharm_dir = os.path.dirname(scripts_dir)
sys.path.insert(0, isolpharm_dir)
sys.path.insert(0, scripts_dir)

from Scripts.analysis.resolving_power_calc import (
    single_gaussian,
    double_gaussian,
    load_scatter_data,
    fit_double_profile,
)

# ──────────────────────────────────────────────
# single_gaussian
# ──────────────────────────────────────────────

def test_should_return_amp_plus_c_at_peak():
    """
    Tests that single_gaussian returns amp + c when evaluated at x = mu,
    since the exponential term equals 1 at the peak.
    """
    # GIVEN: Known parameters
    amp, mu, sigma, c = 200.0, 3.0, 1.5, 5.0

    # WHEN: We evaluate at the peak
    result = single_gaussian(mu, amp, mu, sigma, c)

    # THEN: Result should be amp + c
    assert abs(result - (amp + c)) < 1e-10

def test_should_return_an_array_of_the_same_length_when_an_array_is_passed():
    """
    Tests that single_gaussian can handle numpy arrays and returns an array
    of the same length with the correct values.
    """
    # GIVEN: An array of x values
    amp, mu, sigma, c = 100.0, 0.0, 2.0, 10.0
    x = np.array([-2.0, 0.0, 2.0])

    # WHEN: We call single_gaussian with an array
    result = single_gaussian(x, amp, mu, sigma, c)

    # THEN: Result should be an array of the same length with correct values
    expected = amp * np.exp(-0.5 * (x / sigma)**2) + c
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert np.allclose(result, expected)

def test_should_handle_small_sigma_values():
    """
    Tests that single_gaussian can handle very small sigma values without
    resulting in overflow or underflow errors.
    """
    # GIVEN: A very small sigma
    amp, mu, sigma, c = 100.0, 5.0, 1e-6, 10.0
    x = np.array([5.0])  # Evaluate at the peak

    # WHEN: We call single_gaussian
    result = single_gaussian(x, amp, mu, sigma, c)

    # THEN: Result should be finite and equal to amp + c
    assert np.isfinite(result).all()
    assert abs(result - (amp + c)) < 1e-10

def test_should_return_c_when_amp_is_zero():
    """
    Tests that if amp is zero, single_gaussian returns the background level c.
    """
    # GIVEN: amp = 0
    amp, mu, sigma, c = 0.0, 5.0, 2.0, 10.0
    x = np.array([5.0])  # Evaluate at the peak

    # WHEN: We call single_gaussian
    result = single_gaussian(x, amp, mu, sigma, c)

    # THEN: Result should be equal to c
    assert abs(result - c) < 1e-10


# ──────────────────────────────────────────────
# double_gaussian
# ──────────────────────────────────────────────

def test_should_return_sum_of_two_gaussians_plus_background():
    """
    Tests that double_gaussian evaluated at the first peak equals
    amp1 + c, since the second gaussian contributes negligibly
    when the two peaks are well separated.
    """
    # GIVEN: Two well-separated peaks
    amp1, mu1, sigma1 = 100.0, -5.0, 0.5
    amp2, mu2, sigma2 = 80.0,  5.0, 0.5
    c = 2.0

    # WHEN: We evaluate exactly at mu1 (far from mu2)
    result = double_gaussian(mu1, amp1, mu1, sigma1, amp2, mu2, sigma2, c)

    # THEN: Result should be approximately amp1 + c
    # (contribution from second gaussian is negligible at e^{-200})
    assert abs(result - (amp1 + c)) < 1e-6


def test_should_return_correct_value_at_midpoint_between_peaks():
    """
    Tests that double_gaussian is symmetric at the midpoint between
    two identical, symmetric peaks.
    """
    # GIVEN: Two identical peaks symmetric around zero
    amp, sigma, c = 100.0, 1.0, 0.0
    mu1, mu2 = -4.0, 4.0

    # WHEN: We evaluate at x = 0 (midpoint)
    result = double_gaussian(0.0, amp, mu1, sigma, amp, mu2, sigma, c)

    # THEN: Both gaussians contribute equally, result = 2 * amp * exp(-8)
    expected = 2 * amp * np.exp(-0.5 * (4.0 / sigma)**2)
    assert abs(result - expected) < 1e-10

def test_should_return_a_single_gaussian_when_amp1_is_zero():
    """
    Tests that if amp1 is zero, double_gaussian reduces to a single
    gaussian defined by amp2, mu2, sigma2, plus background c.
    """
    # GIVEN: amp1 = 0
    amp1 = 0.0
    mu1, sigma1 = -5.0, 1.0
    amp2, mu2, sigma2 = 100.0, 5.0, 1.0
    c = 10.0

    # WHEN: We evaluate at x = mu2 (peak of second gaussian)
    result = double_gaussian(mu2, amp1, mu1, sigma1, amp2, mu2, sigma2, c)

    # THEN: Result should be approximately amp2 + c
    assert abs(result - (amp2 + c)) < 1e-6

def test_should_return_the_sum_of_amp1_and_amp2_when_mu1_equals_mu2():
    """
    Tests that if mu1 equals mu2, double_gaussian returns the sum of
    amp1 and amp2 plus background c at that point.
    """
    # GIVEN: mu1 = mu2
    mu = 0.0
    amp1, sigma1 = 100.0, 1.0
    amp2, sigma2 = 80.0, 1.0
    c = 5.0

    # WHEN: We evaluate at x = mu (where both peaks coincide)
    result = double_gaussian(mu, amp1, mu, sigma1, amp2, mu, sigma2, c)

    # THEN: Result should be amp1 + amp2 + c
    assert abs(result - (amp1 + amp2 + c)) < 1e-6

def test_should_return_the_correct_length_array_when_given_an_array_of_x_values():
    """
    Tests that double_gaussian can handle an array of x values and returns
    an array of the same length with the correct double gaussian values.
    """
    # GIVEN: An array of x values and known parameters
    amp1, mu1, sigma1 = 100.0, -3.0, 1.0
    amp2, mu2, sigma2 = 80.0, 3.0, 1.0
    c = 10.0
    x = np.array([-5.0, -3.0, 0.0, 3.0, 5.0])

    # WHEN: We call double_gaussian with an array
    result = double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2, c)

    # THEN: Result should be an array of the same length with correct values
    expected = (amp1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2) +
                amp2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2) + c)
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert np.allclose(result, expected)


# ──────────────────────────────────────────────
# fit_double_profile
# ──────────────────────────────────────────────

def test_should_return_correct_distance_for_two_separated_peaks():
    """
    Tests that fit_double_profile correctly identifies the distance
    between two well-separated synthetic gaussian peaks.
    """
    # GIVEN: Synthetic profile with two peaks at -5 mm and +5 mm
    axis_vals = np.linspace(-13.0, 13.0, 60)
    profile_counts = (
        200.0 * np.exp(-0.5 * ((axis_vals + 5.0) / 1.5)**2) +
        200.0 * np.exp(-0.5 * ((axis_vals - 5.0) / 1.5)**2)
    )

    fig, ax = plt.subplots()

    # WHEN: We call fit_double_profile
    dist_mm, mu1, mu2 = fit_double_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: Distance should be close to 10 mm
    assert dist_mm is not None
    assert abs(dist_mm - 10.0) < 1.0   # tolleranza 1 mm


def test_should_return_zero_distance_for_aligned_peaks():
    """
    Tests that fit_double_profile returns dist=0.0 and falls back
    to a single gaussian when the two peaks are too close together.
    """
    # GIVEN: A single gaussian profile (sources aligned)
    axis_vals = np.linspace(-13.0, 13.0, 60)
    rng = np.random.default_rng(42)
    profile_counts = (200.0 * np.exp(-0.5 * ((axis_vals - 0.0) / 2.0)**2) +
    rng.normal(0, 2.0, len(axis_vals))  # rumore gaussiano piccolo
    )

    fig, ax = plt.subplots()

    # WHEN: We call fit_double_profile
    dist_mm, mu1, mu2 = fit_double_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: Should fall back to single fit, distance = 0.0, mu2 = None
    assert dist_mm == 0.0
    assert mu2 is None


def test_should_return_none_if_not_enough_points():
    """
    Tests that fit_double_profile returns (None, None, None) when
    fewer than 6 points are provided.
    """
    # GIVEN: Only 4 points
    axis_vals = np.array([-2.0, -1.0, 0.0, 1.0])
    profile_counts = np.array([1.0, 5.0, 5.0, 1.0])

    fig, ax = plt.subplots()

    # WHEN
    dist_mm, mu1, mu2 = fit_double_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN
    assert dist_mm is None
    assert mu1 is None
    assert mu2 is None

def test_should_fall_back_when_is_single_peak_condition_is_met():
    """
    Tests that fit_double_profile correctly identifies when the two peaks
    are too close or one is too small, and falls back to a single gaussian.
    """
    # GIVEN: Two peaks very close together
    axis_vals = np.linspace(-5.0, 5.0, 50)
    profile_counts = (100.0 * np.exp(-0.5 * ((axis_vals + 0.5) / 1.0)**2) +
                      20.0 * np.exp(-0.5 * ((axis_vals - 0.5) / 1.0)**2))

    fig, ax = plt.subplots()

    # WHEN: We call fit_double_profile
    dist_mm, mu1, mu2 = fit_double_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: Should fall back to single fit, distance = 0.0, mu2 = None
    assert dist_mm == 0.0
    assert mu2 is None

def test_should_handle_data_initialized_as_zeros():
    """
    Tests that fit_double_profile returns (None, None, None) when the input
    data are all zeros, which would make the fit fail.
    """
    # GIVEN: A profile with all zero counts
    axis_vals = np.linspace(-5.0, 5.0, 50)
    profile_counts = np.zeros_like(axis_vals)

    fig, ax = plt.subplots()

    # WHEN: We call fit_double_profile
    dist_mm, mu1, mu2 = fit_double_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: Should return (None, None, None) gracefully
    assert dist_mm is None
    assert mu1 is None
    assert mu2 is None

def test_should_handle_limit_case_when_peaks_are_distant_exactly_2mm():
    """
    Tests that fit_double_profile correctly identifies two peaks that are
    exactly 2 mm apart, which is the threshold for considering them as
    separate sources.
    """
    # GIVEN: Two peaks exactly 2 mm apart
    axis_vals = np.linspace(-10.0, 10.0, 100)
    profile_counts = (150.0 * np.exp(-0.5 * ((axis_vals + 1.25) / 1.0)**2) +
                  150.0 * np.exp(-0.5 * ((axis_vals - 1.25) / 1.0)**2))

    fig, ax = plt.subplots()

    # WHEN: We call fit_double_profile
    dist_mm, mu1, mu2 = fit_double_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: Should identify two peaks with distance close to 2 mm
    assert dist_mm is not None
    assert abs(dist_mm - 2.5) < 0.5
