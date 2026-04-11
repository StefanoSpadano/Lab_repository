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

def test_should_handle_small_sigma_without_error():
    """
    Tests that the gaussian function can handle very small sigma values
    without resulting in overflow or underflow errors.
    """
    # GIVEN: A very small sigma
    amp, mu, sigma, c = 100.0, 5.0, 1e-6, 10.0
    x = np.array([5.0])  # Evaluate at the peak

    # WHEN: We call the gaussian function
    result = gaussian(x, amp, mu, sigma, c)

    # THEN: Result should still be finite and equal to amp + c
    assert np.isfinite(result).all()
    assert abs(result - (amp + c)) < 1e-10

def test_should_handle_negative_c_value():
    """
    Tests that the gaussian function can handle a negative background (c)
    without producing incorrect results.
    """
    # GIVEN: A negative c value
    amp, mu, sigma, c = 100.0, 5.0, 2.0, -20.0
    x = np.array([5.0])  # Evaluate at the peak

    # WHEN: We call the gaussian function
    result = gaussian(x, amp, mu, sigma, c)

    # THEN: Result should be amp + c (which is 80 in this case)
    assert abs(result - (amp + c)) < 1e-10

def test_should_handle_array_of_x_values():
    """
    Tests that the gaussian function can handle an array of x values and
    returns an array of corresponding gaussian values.
    """
    # GIVEN: An array of x values and known parameters
    amp, mu, sigma, c = 100.0, 5.0, 2.0, 10.0
    x = np.array([3.0, 5.0, 7.0])  # Evaluate at different points

    # WHEN: We call the gaussian function
    result = gaussian(x, amp, mu, sigma, c)

    # THEN: Result should be an array of gaussian values
    expected = amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + c
    np.testing.assert_array_almost_equal(result, expected)


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

def test_should_handle_missing_file_in_both_paths(tmp_path):
    """
    Tests that if neither the primary nor the alternative JSON file exists,
    the function returns (None, None) without crashing.
    """
    # GIVEN: No JSON files in either location

    # WHEN: We call the function
    x, y = load_scatter_data(str(tmp_path))

    # THEN: Should return (None, None)
    assert x is None
    assert y is None

def test_should_handle_secondary_json_format_with_scatter_data(tmp_path):
    """
    Tests that scatter data is correctly loaded when the JSON contains
    a 'scatter_data' key with 'xyclus_x' and 'xyclus_y' inside it.
    """
    # GIVEN: A JSON file in the secondary format
    hist_dir = tmp_path / "histograms"
    hist_dir.mkdir()
    fake_data = {"scatter_data": {"xyclus_x": [1.0, 2.0, 3.0], "xyclus_y": [4.0, 5.0, 6.0]}}
    json_path = hist_dir / "scatter_data.json"
    json_path.write_text(json.dumps(fake_data))

    # WHEN: We call the function
    x, y = load_scatter_data(str(tmp_path))

    # THEN: Arrays are correctly loaded
    np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y, np.array([4.0, 5.0, 6.0]))

def test_should_handle_malformed_json(tmp_path):
    """
    Tests that if the JSON file is malformed, the function handles the exception
    and returns (None, None) without crashing.
    """
    # GIVEN: A malformed JSON file
    hist_dir = tmp_path / "histograms"
    hist_dir.mkdir()
    json_path = hist_dir / "scatter_data.json"
    json_path.write_text("This is not a valid JSON")

    # WHEN: We call the function
    x, y = load_scatter_data(str(tmp_path))

    # THEN: Should return (None, None) and not raise an exception
    assert x is None
    assert y is None


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
    assert abs(fwhm - expected_fwhm) < 0.5   # tolerance 0.5 mm


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

def test_should_return_none_if_data_are_all_zeros():
    """
    Tests that fit_profile returns (None, None) when the input data
    are all zeros, which would make the fit fail.
    """
    # GIVEN: A profile with all zero counts
    axis_vals = np.linspace(-5.0, 5.0, 50)
    profile_counts = np.zeros_like(axis_vals)

    fig, ax = plt.subplots()

    # WHEN: We call fit_profile
    fwhm, fwhm_err = fit_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: Should return None, None gracefully
    assert fwhm is None
    assert fwhm_err is None

def test_should_warn_about_pixel_locking_for_very_narrow_peaks():
    """
    Tests that fit_profile identifies when the fitted FWHM is below 0.5 mm
    and returns it with a warning status.
    """
    # GIVEN: A very narrow gaussian profile
    amp, mu, sigma, c = 500.0, 0.0, 0.1, 0.0
    axis_vals = np.linspace(-1.0, 1.0, 100)
    profile_counts = gaussian(axis_vals, amp, mu, sigma, c)

    fig, ax = plt.subplots()

    # WHEN: We call fit_profile
    fwhm, fwhm_err = fit_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()

    # THEN: FWHM should be below 0.5 mm and status should indicate pixel locking
    assert fwhm is not None
    assert fwhm < 0.5

def test_should_converge_even_for_noisy_gaussian_data():
    """
    Tests that fit_profile can still return a reasonable FWHM estimate
    even when the input data are noisy, as long as there are enough points.
    """
    # GIVEN: A noisy gaussian profile
    amp, mu, sigma, c = 500.0, 0.0, 3.0, 0.0
    axis_vals = np.linspace(-15.0, 15.0, 60)
    clean_counts = gaussian(axis_vals, amp, mu, sigma, c)
    noise = np.random.normal(0, 20.0, size=clean_counts.shape)
    profile_counts = clean_counts + noise

    fig, ax = plt.subplots()

    # WHEN: We call fit_profile
    fwhm, fwhm_err = fit_profile(axis_vals, profile_counts, "Test", ax)
    plt.close()