import os
import sys
import json
import pytest
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)          # Scripts/
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)
sys.path.insert(0, scripts_dir)    


from Scripts.analysis.psf_calc import gaussian, load_scatter_data, fit_profile

# ──────────────────────────────────────────────
# gaussian
# ──────────────────────────────────────────────

def test_should_return_correct_gaussian_value_at_peak():
    """
    Tests that the gaussian function returns amp + c at x = mu,
    since the exponential term is 1 when x == mu.
    """
    # GIVEN: Known parameters
    amp, mu, sigma, c = 100.0, 5.0, 2.0, 10.0

    # WHEN: We evaluate the gaussian exactly at the peak
    result = gaussian(mu, amp, mu, sigma, c)

    # THEN: Result should be amp + c (exponential = 1 at peak)
    assert abs(result - (amp + c)) < 1e-10


# ──────────────────────────────────────────────
# load_scatter_data
# ──────────────────────────────────────────────

def test_should_load_scatter_data_from_primary_json_format(tmp_path):
    """
    Tests that scatter data is correctly loaded when the JSON contains
    top-level keys 'xyclus_x' and 'xyclus_y'.
    """
    # GIVEN: A JSON file in the primary format
    hist_dir = tmp_path / "histograms"
    hist_dir.mkdir()
    fake_data = {"xyclus_x": [1.0, 2.0, 3.0], "xyclus_y": [4.0, 5.0, 6.0]}
    json_path = hist_dir / "scatter_data.json"
    json_path.write_text(json.dumps(fake_data))

    # WHEN: We call the function
    x, y = load_scatter_data(str(tmp_path))

    # THEN: Arrays are correctly loaded
    np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y, np.array([4.0, 5.0, 6.0]))


# ──────────────────────────────────────────────
# fit_profile
# ──────────────────────────────────────────────

def test_should_return_reasonable_fwhm_for_clean_gaussian_data():
    """
    Tests that fit_profile returns a FWHM close to the expected value
    when given synthetic data generated from a known gaussian.
    """
    # GIVEN: Synthetic gaussian profile with known parameters
    amp, mu, sigma, c = 500.0, 0.0, 3.0, 0.0
    axis_vals = np.linspace(-15.0, 15.0, 60)
    profile_counts = gaussian(axis_vals, amp, mu, sigma, c)

    fig, ax = plt.subplots()

    # WHEN: We call fit_profile
    fwhm, fwhm_err = fit_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: FWHM should be close to 2.355 * sigma = 7.065 mm
    expected_fwhm = 2.355 * sigma
    assert fwhm is not None
    assert abs(fwhm - expected_fwhm) < 0.5   # tolleranza 0.5 mm


def test_should_return_none_if_not_enough_points():
    """
    Tests that fit_profile returns (None, None) when the ROI
    contains fewer than 4 points, making the fit impossible.
    """
    # GIVEN: A profile with very few points
    axis_vals = np.array([-1.0, 0.0, 1.0])
    profile_counts = np.array([1.0, 5.0, 1.0])

    fig, ax = plt.subplots()

    # WHEN: We call fit_profile
    fwhm, fwhm_err = fit_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: Should return None, None gracefully
    assert fwhm is None
    assert fwhm_err is None