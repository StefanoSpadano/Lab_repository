import os
import sys
import json
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)

from Scripts.analysis.fit_scipy_step import (
    crystal_ball,
    step_function,
    full_model,
    rebin_histogram,
    load_histogram_data,
    perform_step_fit,
)


# ──────────────────────────────────────────────
# crystal_ball
# ──────────────────────────────────────────────

def test_should_return_N_at_peak_for_valid_parameters():
    """
    Tests that crystal_ball returns N when evaluated at x = mu,
    since the Gaussian core evaluates to exp(0) = 1 at the peak.
    """
    # GIVEN: Known parameters with x at the peak
    N, mu, sigma, alpha, n = 1000.0, 5000.0, 200.0, 1.5, 2.0
    x = np.array([mu])

    # WHEN: We evaluate crystal_ball at the peak
    result = crystal_ball(x, N, mu, sigma, alpha, n)

    # THEN: Result should equal N
    assert abs(result[0] - N) < 1e-6


def test_should_return_zeros_when_sigma_is_non_positive():
    """
    Tests that crystal_ball returns an array of zeros when sigma <= 0,
    as required by the guard clause at the start of the function.
    """
    # GIVEN: sigma = 0 and a range of x values
    x = np.array([4800.0, 5000.0, 5200.0])
    N, mu, sigma, alpha, n = 1000.0, 5000.0, 0.0, 1.5, 2.0

    # WHEN: We call crystal_ball with sigma = 0
    result = crystal_ball(x, N, mu, sigma, alpha, n)

    # THEN: All values should be zero
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_should_treat_negative_alpha_as_its_absolute_value():
    """
    Tests that crystal_ball treats negative alpha as abs(alpha),
    so passing -alpha and +alpha should yield identical results.
    """
    # GIVEN: Same parameters with alpha positive and negative
    x = np.linspace(4000.0, 6000.0, 50)
    N, mu, sigma, n = 1000.0, 5000.0, 200.0, 2.0

    # WHEN: We call crystal_ball with positive and negative alpha
    result_pos = crystal_ball(x, N, mu, sigma, 1.5, n)
    result_neg = crystal_ball(x, N, mu, sigma, -1.5, n)

    # THEN: Results should be identical
    np.testing.assert_array_almost_equal(result_pos, result_neg)


def test_should_return_finite_non_negative_values_over_full_adc_range():
    """
    Tests that crystal_ball returns finite, non-negative values for a full ADC
    range typical of a detector spectrum, with no NaN or Inf values.
    """
    # GIVEN: A typical ADC range and physically plausible parameters
    x = np.linspace(0.0, 16000.0, 500)
    N, mu, sigma, alpha, n = 2000.0, 5000.0, 200.0, 1.5, 3.0

    # WHEN: We call crystal_ball
    result = crystal_ball(x, N, mu, sigma, alpha, n)

    # THEN: All values should be finite and non-negative
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)


# ──────────────────────────────────────────────
# step_function
# ──────────────────────────────────────────────

def test_should_return_half_S_at_the_step_position():
    """
    Tests that step_function returns S * 0.5 when evaluated at x = mu_step,
    since erfc(0) = 1, so the result is S * 0.5 * 1 = S * 0.5.
    """
    # GIVEN: Known parameters, evaluating at the step centre
    S, mu_step, sigma_step = 400.0, 4700.0, 150.0
    x = np.array([mu_step])

    # WHEN: We evaluate step_function at x = mu_step
    result = step_function(x, S, mu_step, sigma_step)

    # THEN: Result should equal S * 0.5
    assert abs(result[0] - S * 0.5) < 1e-6


def test_should_return_approximately_S_far_to_the_left_of_step():
    """
    Tests that step_function approaches S when x is much smaller than mu_step,
    since erfc(-inf) = 2 and the result tends to S * 0.5 * 2 = S.
    """
    # GIVEN: x very far to the left of the step (50 sigma away)
    S, mu_step, sigma_step = 400.0, 5000.0, 100.0
    x = np.array([mu_step - 5000.0])

    # WHEN: We call step_function
    result = step_function(x, S, mu_step, sigma_step)

    # THEN: Result should be very close to S
    assert abs(result[0] - S) < 1e-6


def test_should_return_approximately_zero_far_to_the_right_of_step():
    """
    Tests that step_function approaches 0 when x is much larger than mu_step,
    since erfc(+inf) = 0, so the result tends to S * 0.5 * 0 = 0.
    """
    # GIVEN: x very far to the right of the step (50 sigma away)
    S, mu_step, sigma_step = 400.0, 5000.0, 100.0
    x = np.array([mu_step + 5000.0])

    # WHEN: We call step_function
    result = step_function(x, S, mu_step, sigma_step)

    # THEN: Result should be approximately zero
    assert abs(result[0]) < 1e-6


def test_should_return_zeros_everywhere_when_S_is_zero():
    """
    Tests that if S is zero, step_function returns zero for all x values,
    since the step amplitude is zero.
    """
    # GIVEN: S = 0
    S, mu_step, sigma_step = 0.0, 5000.0, 150.0
    x = np.array([4000.0, 5000.0, 6000.0])

    # WHEN: We call step_function
    result = step_function(x, S, mu_step, sigma_step)

    # THEN: All values should be zero
    np.testing.assert_array_almost_equal(result, np.zeros_like(x))


# ──────────────────────────────────────────────
# full_model
# ──────────────────────────────────────────────

def test_should_equal_sum_of_components_at_every_point():
    """
    Tests that full_model returns exactly crystal_ball + step_function + c
    for an array of x values, verifying the additive composition.
    """
    # GIVEN: Known parameters and an array of x values
    x = np.linspace(3000.0, 7000.0, 100)
    N, mu, sigma, alpha, n = 1000.0, 5000.0, 200.0, 1.5, 2.0
    S, mu_step, sigma_step, c = 300.0, 4700.0, 200.0, 15.0

    # WHEN: We compute full_model and the direct component sum separately
    result = full_model(x, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c)
    expected = (crystal_ball(x, N, mu, sigma, alpha, n) +
                step_function(x, S, mu_step, sigma_step) + c)

    # THEN: They should be exactly equal
    np.testing.assert_array_almost_equal(result, expected)


def test_should_reduce_to_crystal_ball_plus_c_when_S_is_zero():
    """
    Tests that full_model reduces to crystal_ball + c when S = 0,
    i.e. when the step component is absent.
    """
    # GIVEN: S = 0, so step_function contributes nothing
    x = np.linspace(3000.0, 7000.0, 80)
    N, mu, sigma, alpha, n = 1000.0, 5000.0, 200.0, 1.5, 2.0
    S, mu_step, sigma_step, c = 0.0, 4700.0, 200.0, 20.0

    # WHEN: We compute full_model
    result = full_model(x, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c)
    expected = crystal_ball(x, N, mu, sigma, alpha, n) + c

    # THEN: Results should match
    np.testing.assert_array_almost_equal(result, expected)


def test_should_add_constant_c_uniformly_across_all_x():
    """
    Tests that the constant c is an additive offset, so the difference between
    two calls with different c values equals the difference in c at every point.
    """
    # GIVEN: Two calls with different constants, same other parameters
    x = np.linspace(3000.0, 7000.0, 80)
    N, mu, sigma, alpha, n = 1000.0, 5000.0, 200.0, 1.5, 2.0
    S, mu_step, sigma_step = 300.0, 4700.0, 200.0
    c1, c2 = 10.0, 50.0

    # WHEN: We compute full_model with two different c values
    result_c1 = full_model(x, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c1)
    result_c2 = full_model(x, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c2)

    # THEN: The difference should be c2 - c1 uniformly everywhere
    np.testing.assert_array_almost_equal(result_c2 - result_c1, np.full_like(x, c2 - c1))


def test_should_return_only_c_when_N_and_S_are_both_zero():
    """
    Tests that full_model returns the constant c everywhere when N = 0 and S = 0,
    since both the crystal ball and the step amplitudes vanish.
    """
    # GIVEN: N = 0 and S = 0
    x = np.array([3000.0, 5000.0, 7000.0])
    N, mu, sigma, alpha, n = 0.0, 5000.0, 200.0, 1.5, 2.0
    S, mu_step, sigma_step, c = 0.0, 4700.0, 200.0, 25.0

    # WHEN: We call full_model
    result = full_model(x, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c)

    # THEN: All values should equal c
    np.testing.assert_array_almost_equal(result, np.full_like(x, c))


# ──────────────────────────────────────────────
# rebin_histogram
# ──────────────────────────────────────────────

def test_should_sum_counts_in_pairs_and_halve_bins_for_rebin_factor_two():
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


def test_should_return_unchanged_data_when_rebin_factor_is_one():
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


def test_should_return_empty_arrays_when_rebin_factor_exceeds_number_of_bins():
    """
    Tests that if rebin_factor is larger than the number of bins, empty arrays are returned.
    """
    # GIVEN: A histogram with 3 bins
    bins = np.array([0.0, 1.0, 2.0, 3.0])   # 4 edges -> 3 bins
    counts = np.array([10.0, 20.0, 30.0])

    # WHEN: We rebin by a factor of 4 (more than number of bins)
    new_bins, new_counts = rebin_histogram(bins, counts, rebin_factor=4)

    # THEN: Empty arrays should be returned
    assert len(new_bins) == 1
    assert len(new_counts) == 0


def test_should_truncate_excess_bins_when_rebin_factor_does_not_divide_evenly():
    """
    Tests that if rebin_factor does not divide the number of bins evenly,
    the trailing excess bins are silently dropped before rebinning.
    """
    # GIVEN: A histogram with 5 bins
    bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])   # 6 edges -> 5 bins
    counts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # WHEN: We rebin by a factor of 2 (which does not divide 5 evenly)
    new_bins, new_counts = rebin_histogram(bins, counts, rebin_factor=2)

    # THEN: The last bin is dropped and the first four bins are rebinned in pairs
    np.testing.assert_array_equal(new_counts, np.array([30.0, 70.0]))   # (10+20) and (30+40)
    np.testing.assert_array_equal(new_bins, np.array([0.0, 2.0, 4.0]))


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
    (hist_dir / "total_charge.json").write_text(json.dumps(fake_data))

    # WHEN: We call the function
    bins, counts = load_histogram_data(str(tmp_path), "total_charge")

    # THEN: Arrays are correctly loaded as numpy arrays
    np.testing.assert_array_equal(bins, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(counts, np.array([10.0, 20.0]))


def test_should_return_none_if_json_file_not_found(tmp_path):
    """
    Tests that if the expected JSON file is not found, the function returns (None, None).
    """
    # GIVEN: No JSON file in the expected directory structure

    # WHEN: We call the function with a missing file
    bins, counts = load_histogram_data(str(tmp_path), "total_charge")

    # THEN: Both return values should be None
    assert bins is None
    assert counts is None


def test_should_handle_json_in_list_format(tmp_path):
    """
    Tests that a JSON file in list format [[bins], [counts]]
    is correctly loaded as the secondary supported format.
    """
    # GIVEN: A JSON file as a list of two lists [bins_list, counts_list]
    hist_dir = tmp_path / "histograms"
    hist_dir.mkdir()
    fake_data = [[0.0, 1.0, 2.0], [10.0, 20.0]]
    (hist_dir / "total_charge.json").write_text(json.dumps(fake_data))

    # WHEN: We call the function
    bins, counts = load_histogram_data(str(tmp_path), "total_charge")

    # THEN: Arrays are correctly loaded
    np.testing.assert_array_equal(bins, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(counts, np.array([10.0, 20.0]))


def test_should_return_none_for_malformed_json(tmp_path):
    """
    Tests that if the JSON file is malformed, the function handles the exception
    and returns (None, None) without crashing.
    """
    # GIVEN: A malformed JSON file
    hist_dir = tmp_path / "histograms"
    hist_dir.mkdir()
    (hist_dir / "total_charge.json").write_text("This is not valid JSON {{")

    # WHEN: We call the function
    bins, counts = load_histogram_data(str(tmp_path), "total_charge")

    # THEN: Both return values should be None
    assert bins is None
    assert counts is None


def test_should_return_none_for_json_with_unexpected_structure(tmp_path):
    """
    Tests that if the JSON file does not contain the expected keys or structure,
    the function returns (None, None) without crashing.
    """
    # GIVEN: A JSON file with an unrecognised structure (no "bins" / "counts" keys)
    hist_dir = tmp_path / "histograms"
    hist_dir.mkdir()
    (hist_dir / "total_charge.json").write_text(json.dumps({"unexpected": [1, 2, 3]}))

    # WHEN: We call the function
    bins, counts = load_histogram_data(str(tmp_path), "total_charge")

    # THEN: Both return values should be None
    assert bins is None
    assert counts is None


# ──────────────────────────────────────────────
# perform_step_fit
# ──────────────────────────────────────────────

def test_should_return_none_when_no_counts_fall_in_search_roi(tmp_path):
    """
    Tests that perform_step_fit returns None when no bins fall in the
    search ROI (3000–8000 ADC), which triggers the early-exit guard.
    """
    # GIVEN: A histogram whose bins lie entirely below the search ROI
    bins = np.linspace(0.0, 2000.0, 201)   # all bin centres below 3000 ADC
    counts = np.ones(200) * 100.0

    # WHEN: We call perform_step_fit
    result = perform_step_fit(bins, counts, "TestLabel", str(tmp_path))

    # THEN: Should return None since there are no points in the search ROI
    assert result is None


def test_should_return_none_when_too_few_points_in_fit_window(tmp_path):
    """
    Tests that perform_step_fit returns None when the fit window around the
    detected peak contains fewer than 10 data points after rebinning.
    """
    # GIVEN: A very coarse histogram — 8 bins of 2000 ADC each; after rebin by 2
    # the bin width becomes 4000 ADC, leaving only 1 bin in the 2000-ADC fit window.
    bins = np.linspace(0.0, 16000.0, 9)
    counts = np.array([0.0, 0.0, 500.0, 200.0, 0.0, 0.0, 0.0, 0.0])

    # WHEN: We call perform_step_fit
    result = perform_step_fit(bins, counts, "TestLabel", str(tmp_path))

    # THEN: Should return None due to insufficient points in the fit window
    assert result is None


def test_should_return_dict_with_expected_keys_for_a_well_defined_spectrum(tmp_path):
    """
    Tests that perform_step_fit returns a dictionary with all expected result keys
    when given a clean synthetic crystal ball + step + constant spectrum whose peak
    sits well inside the search ROI.
    """
    # GIVEN: A synthetic spectrum with a peak at 5000 ADC (centre of search ROI)
    bins = np.linspace(0.0, 16000.0, 8001)   # 8000 bins, 2 ADC each
    centers = 0.5 * (bins[:-1] + bins[1:])
    N, mu, sigma, alpha, n = 3000.0, 5000.0, 150.0, 1.5, 2.0
    S, mu_step, sigma_step, c = 500.0, 4700.0, 150.0, 10.0
    counts = full_model(centers, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c)

    # WHEN: We call perform_step_fit
    result = perform_step_fit(bins, counts, "TestLabel", str(tmp_path))

    # THEN: Should return a dict with all expected fit result keys
    assert result is not None
    assert "res" in result
    assert "chi2" in result
    assert "mu" in result
    assert "sigma" in result
    assert "amp" in result
    assert "step_amp" in result
    assert "const" in result


def test_should_recover_peak_position_within_tolerance_for_synthetic_spectrum(tmp_path):
    """
    Tests that perform_step_fit recovers the true peak position mu within ±500 ADC
    when given a noise-free synthetic crystal ball spectrum centred at 5000 ADC.
    """
    # GIVEN: A noiseless spectrum with a well-defined peak at 5000 ADC
    bins = np.linspace(0.0, 16000.0, 8001)
    centers = 0.5 * (bins[:-1] + bins[1:])
    N, mu, sigma, alpha, n = 3000.0, 5000.0, 150.0, 1.5, 2.0
    S, mu_step, sigma_step, c = 500.0, 4700.0, 150.0, 10.0
    counts = full_model(centers, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c)

    # WHEN: We call perform_step_fit
    result = perform_step_fit(bins, counts, "TestLabel", str(tmp_path))

    # THEN: Fitted peak position should be within 500 ADC of the true value
    assert result is not None
    assert abs(result["mu"] - mu) < 500
