import os
import sys
import json
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)

from Scripts.analysis.fit_scipy_gmm import (
    calculate_resolution,
    rebin_histogram,
    load_histogram_data,
)

# ──────────────────────────────────────────────
# calculate_resolution
# ──────────────────────────────────────────────

def test_should_calculate_correct_resolution():
    """
    Tests that the resolution is correctly computed as (FWHM / mu) * 100,
    where FWHM = 2.355 * sigma.
    """
    # GIVEN: A photopeak with known mu and sigma
    mu = 5000.0
    sigma = 200.0

    # WHEN: We call the function
    result = calculate_resolution(mu, sigma)

    # THEN: Resolution = (2.355 * 200 / 5000) * 100 = 9.42 %
    expected = (2.355 * sigma / mu) * 100
    assert abs(result - expected) < 1e-6

def test_should_return_zero_resolution_for_zero_mu():
    """
    Tests that if mu is zero, the function returns 0 to avoid division by zero.
    """
    # GIVEN: A photopeak with mu = 0
    mu = 0.0
    sigma = 200.0

    # WHEN: We call the function
    result = calculate_resolution(mu, sigma)

    # THEN: Resolution should be 0
    assert result == 0.0

def test_should_return_zero_resolution_for_zero_sigma():
    """
    Tests that if sigma is zero, the function returns 0 since FWHM would be zero.
    """
    # GIVEN: A photopeak with sigma = 0
    mu = 5000.0
    sigma = 0.0

    # WHEN: We call the function
    result = calculate_resolution(mu, sigma)

    # THEN: Resolution should be 0 and not NaN or inf
    assert result == 0.0
    assert not np.isnan(result)
    assert not np.isinf(result)


# ──────────────────────────────────────────────
# rebin_histogram
# ──────────────────────────────────────────────

def test_should_sum_counts_and_reduce_bins_when_rebinning():
    """
    Tests that with rebin_factor=2, adjacent counts are summed
    and the number of bins is halved accordingly.
    """
    # GIVEN: A simple histogram with 4 bins
    bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0])   # 5 edges -> 4 bins
    counts = np.array([10.0, 20.0, 30.0, 40.0])

    # WHEN: We rebin by a factor of 2
    new_bins, new_counts = rebin_histogram(bins, counts, rebin_factor=2)

    # THEN: Counts are summed in pairs and bins are halved
    np.testing.assert_array_equal(new_counts, np.array([30.0, 70.0]))
    np.testing.assert_array_equal(new_bins, np.array([0.0, 2.0, 4.0]))

def test_should_check_data_are_not_changed_if_rebin_factor_is_one():
    """
    Tests that if rebin_factor=1, the original bins and counts are returned unchanged.
    """
    # GIVEN: A simple histogram
    bins = np.array([0.0, 1.0, 2.0, 3.0])   # 4 edges -> 3 bins
    counts = np.array([10.0, 20.0, 30.0])

    # WHEN: We rebin by a factor of 1 (no rebinning)
    new_bins, new_counts = rebin_histogram(bins, counts, rebin_factor=1)

    # THEN: The original bins and counts are returned unchanged
    np.testing.assert_array_equal(new_bins, bins)
    np.testing.assert_array_equal(new_counts, counts)

def test_should_return_empty_arrays_if_rebin_factor_exceeds_bins():
    """
    Tests that if rebin_factor is larger than the number of bins, empty arrays are returned.
    """
    # GIVEN: A histogram with 3 bins
    bins = np.array([0.0, 1.0, 2.0, 3.0])   # 4 edges -> 3 bins
    counts = np.array([10.0, 20.0, 30.0])

    # WHEN: We rebin by a factor of 4 (more than number of bins)
    new_bins, new_counts = rebin_histogram(bins, counts, rebin_factor=4)

    # THEN: Empty arrays should be returned
    assert len(new_bins) == 0
    assert len(new_counts) == 0


# ──────────────────────────────────────────────
# load_histogram_data
# ──────────────────────────────────────────────

def test_should_load_histogram_from_dict_format(tmp_path):
    """
    Tests that a JSON file in dict format {"bins": [...], "counts": [...]}
    is correctly loaded and returned as numpy arrays.
    """
    # GIVEN: A JSON file in dict format saved in the expected directory structure
    hist_dir = tmp_path / "histograms"
    hist_dir.mkdir()
    fake_data = {"bins": [0.0, 1.0, 2.0], "counts": [10.0, 20.0]}
    json_path = hist_dir / "total_charge.json"
    json_path.write_text(json.dumps(fake_data))

    # WHEN: We call the function
    bins, counts = load_histogram_data(str(tmp_path), "total_charge")

    # THEN: The arrays are correctly loaded
    np.testing.assert_array_equal(bins, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(counts, np.array([10.0, 20.0]))