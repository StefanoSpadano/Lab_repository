import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


current_dir = os.path.dirname(os.path.abspath(__file__))
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)

from Scripts.analysis.event_processing import collect_all_histograms, load_calibration_dynamic, apply_subtraction_and_plot, elapsed_time

def test_should_calculate_correct_elapsed_time():
    """
    Tests that the difference between timestamps in microseconds
    is correctly converted to seconds.
    """
    # GIVEN: An array of timestamps in microseconds
    # 1000000 microseconds = 1 second. 
    # Difference between min (1.000.000) and max (3.500.000) = 2.500.000 microseconds -> 2.5 seconds
    fake_trig_time = np.array([1000000, 2000000, 3500000])

    # WHEN: We call the function
    result = elapsed_time(fake_trig_time)

    # THEN: The result should be exactly 2.5
    assert result == 2.5

def test_should_return_one_second_for_empty_arrays():
    """
    Tests that if the input array is empty, the function returns 1.0 seconds.
    The 1 second clause is made so that later on is possible to avoid crashes due to division by zero when calculating rates or scaling background.
    """
    # GIVEN: An empty array (no events recorded)
    empty_trig_time = np.array([])

    # WHEN: We call the function
    result = elapsed_time(empty_trig_time)

    # THEN: The result should be the default safety value of 1.0
    assert result == 1.0

# @patch intercepts the plot function and replaces it with a "fake" one (mock_plot)
@patch("Scripts.analysis.event_processing.plot_subtraction_check")
def test_should_correctly_subtract_scaled_background(mock_plot):
    """
    Tests that the background is correctly scaled over time and subtracted
    from the run data.
    """
    # GIVEN: Fake data for both background and amplification lines data
    bins = np.array([0, 1, 2, 3])
    run_counts = np.array([100, 200, 150])
    bkg_counts = np.array([10,  20,  15])

    res_run = {
        "acquisition_time_sec": 10.0,
        "histograms": {
            "total_charge": (bins, run_counts)
        }
    }

    bkg_data = {
        "acquisition_time_sec": 5.0, # Let's say that the background acquisition time is half of the run time
        "histograms": {
            "total_charge": (bins, bkg_counts)
        }
    }

    # WHEN: We call the function
    result = apply_subtraction_and_plot(res_run, bkg_data, "fake_run_path.root")

    # THEN: We check the math
    # The scaling factor should be t_run / t_bkg = 10.0 / 5.0 = 2.0
    # So the scaled background is: [20, 40, 30]
    # Subtraction (run - scaled_bkg): [100-20, 200-40, 150-30] = [80, 160, 120]
    
    net_bins, net_counts = result["histograms"]["total_charge"]
    
    # In order to compare NumPy arrays, we use np.testing.assert_array_equal
    np.testing.assert_array_equal(net_bins, bins)
    np.testing.assert_array_equal(net_counts, np.array([80, 160, 120]))
    
    # Verify that the plot function was called (even if we mocked it)
    mock_plot.assert_called_once()

@patch("Scripts.analysis.event_processing.plot_subtraction_check")
def test_should_skip_if_bins_mismatch(mock_plot):
    """
    Tests that if the bins of the run and background histograms do not match,
    the function should skip subtraction and return the original run data.
    """
    # GIVEN: Fake data with mismatching bins
    run_bins = np.array([0, 1, 2, 3])
    bkg_bins = np.array([0, 2, 4])  # Different binning
    run_counts = np.array([100, 200, 150])
    bkg_counts = np.array([10, 20])

    res_run = {
        "acquisition_time_sec": 10.0,
        "histograms": {
            "total_charge": (run_bins, run_counts)
        }
    }

    bkg_data = {
        "acquisition_time_sec": 5.0,
        "histograms": {
            "total_charge": (bkg_bins, bkg_counts)
        }
    }

    # WHEN: We call the function
    result = apply_subtraction_and_plot(res_run, bkg_data, "fake_run_path.root")

    # THEN: The result should be unchanged (no subtraction)
    net_bins, net_counts = result["histograms"]["total_charge"]
    
    np.testing.assert_array_equal(net_bins, run_bins)
    np.testing.assert_array_equal(net_counts, run_counts)
    
    # Verify that the plot function was NOT called since subtraction was skipped
    mock_plot.assert_not_called()

@patch("Scripts.analysis.event_processing.plot_subtraction_check")
def test_should_prevent_zero_division_for_invalid_bkg_time(mock_plot):
    """
    Tests that if the background acquisition time is zero, the function should
    handle it gracefully and not perform subtraction (to avoid division by zero).
    """
    # GIVEN: Fake data with zero background acquisition time
    bins = np.array([0, 1, 2, 3])
    run_counts = np.array([100, 200, 150])
    bkg_counts = np.array([10, 20, 30])

    res_run = {
        "acquisition_time_sec": 10.0,
        "histograms": {
            "total_charge": (bins, run_counts)
        }
    }

    bkg_data = {
        "acquisition_time_sec": 0.0,  # Invalid background time
        "histograms": {
            "total_charge": (bins, bkg_counts)
        }
    }

    # WHEN: We call the function
    result = apply_subtraction_and_plot(res_run, bkg_data, "fake_run_path.root")

    # THEN: The subtraction still happens but a forced t_bkg = 1.0 is used
    net_bins, net_counts = result["histograms"]["total_charge"]
    
    # The background scaled by 10 and subtracted gives [0, 0, -150]
    expected_net_counts = np.array([0, 0, -150])
    
    np.testing.assert_array_equal(net_bins, bins)
    np.testing.assert_array_equal(net_counts, expected_net_counts)
    
    # Since the subtraction still happens (with a fallback time), the plot function should be called
    mock_plot.assert_called_once()


def test_should_load_valid_calibration_excel(tmp_path):
    """
    Tests that a valid calibration Excel file is loaded correctly,
    replacing commas with dots and skipping unspecified channels.
    """
    # GIVEN: Create a fake DataFrame and save it as an Excel file
    # Insert a channel with a comma (e.g., 1,05) to test the text correction
    fake_data = {
        "Channel": ["0", "15", "63"],
        "Correction": ["1.2", "0,95", "1.05"]
    }
    df = pd.DataFrame(fake_data)
    
    # Save the Excel file in the temporary folder (the name MUST be calibration.xlsx)
    fake_run_dir = tmp_path / "Run_Test"
    fake_run_dir.mkdir()
    excel_path = fake_run_dir / "calibration.xlsx"
    
    # Pandas is used to write the fake file to disk
    df.to_excel(excel_path, sheet_name="Correzioni", index=False)

    # WHEN: Call the function passing the fake directory
    corr_array, calib_source = load_calibration_dynamic(str(fake_run_dir))

    # THEN: Check that the array has been updated correctly
    assert len(corr_array) == 64
    assert corr_array[0] == 1.2
    assert corr_array[15] == 0.95  # The comma should have been handled!
    assert corr_array[63] == 1.05
    
    # A channel not mentioned in the file must remain at the default value 1.0
    assert corr_array[1] == 1.0 
    
    # Check the output string
    assert "calibration.xlsx" in calib_source
    assert "3 ch" in calib_source

def test_should_handle_missing_calibration_file(tmp_path):
    """
    Tests that if the calibration Excel file is missing, the function returns
    the default correction array and an appropriate message.
    """
    # GIVEN: A directory without a calibration.xlsx file
    fake_run_dir = tmp_path / "Run_No_Calib"
    fake_run_dir.mkdir()

    # WHEN: Call the function passing the fake directory
    corr_array, calib_source = load_calibration_dynamic(str(fake_run_dir))

    # THEN: The correction array should be all ones (default)
    assert len(corr_array) == 64
    assert np.all(corr_array == 1.0)
    
    #The test should print the default value "None (Unity)" 
    assert calib_source == "None (Unity)"

def test_should_ignore_invalid_entries_in_calibration_file(tmp_path):
    """
    Tests that if the calibration Excel file contains invalid entries (e.g., non-numeric corrections),
    the function should ignore those entries and keep the default value for those channels.
    """
    # GIVEN: Create a fake DataFrame with some invalid correction values
    fake_data = {
        "Channel": ["0", "15", "63"],
        "Correction": ["1.2", "invalid", "1.05"]  # Channel 15 has an invalid correction
    }
    df = pd.DataFrame(fake_data)
    
    # Save the Excel file in the temporary folder
    fake_run_dir = tmp_path / "Run_Invalid_Calib"
    fake_run_dir.mkdir()
    excel_path = fake_run_dir / "calibration.xlsx"
    
    df.to_excel(excel_path, sheet_name="Correzioni", index=False)

    # WHEN: Call the function passing the fake directory
    corr_array, calib_source = load_calibration_dynamic(str(fake_run_dir))

    # THEN: The valid entries should be loaded, and the invalid one should be ignored
    assert len(corr_array) == 64
    assert corr_array[0] == 1.2
    assert corr_array[15] == 1.0  # Invalid entry should be ignored, default remains
    assert corr_array[63] == 1.05
    
    # Check the output string mentions the valid channel count (2 valid channels)
    assert "calibration.xlsx" in calib_source
    assert "2 ch" in calib_source

@patch("Scripts.analysis.event_processing.uproot.open")
def test_should_process_events_and_generate_histograms(mock_uproot, tmp_path):
    """
    Tests that the function reads the physical data (mocked), applies the thresholds,
    and generates the correct structure of the results.
    """
    # GIVEN: We build fake data arrays as if they were coming from a ROOT file.
    # Event 0: Channels 0 and 1 are on (with signals above the PEDESTAL of 50 and the THR of 75)
    # Event 1: Channel 5 is on
    fake_data = {
        "trigTime": np.array([1000000, 2000000]),
        # We use dtype=object because they are variable-length arrays (ragged arrays)
        "channelID": np.array([[0, 1], [5]], dtype=object),
        "channelDataLG": np.array([[200, 150], [300]], dtype=object) 
    }

    # Using MagicMock we simulate the behavior of 'with uproot.open(..) as f: f["fersTree"].arrays(...)'
    mock_tree = MagicMock()
    mock_tree.arrays.return_value = fake_data
    
    mock_file = MagicMock()
    mock_file.__getitem__.return_value = mock_tree # Simula f["fersTree"]
    
    # This simulates the context manager (the 'with' keyword)
    mock_uproot.return_value.__enter__.return_value = mock_file

    # We build a fake directory just to avoid making the calibration upset
    fake_run_dir = tmp_path / "Run0"
    fake_run_dir.mkdir()
    fake_root_path = fake_run_dir / "Run0.root"

    # WHEN: We call the function (the background is None)
    result = collect_all_histograms(str(fake_root_path))

    # THEN: Control that the final dictionary exists and has the correct keys
    assert "histograms" in result
    assert "total_charge" in result["histograms"]
    assert "maps" in result
    
    # The event 0 had 200 and 150. Subtracted the pedestal (50): (200-50) + (150-50) = 150 + 100 = 250
    # The event 1 had 300. Subtracted the pedestal (50): 300 - 50 = 250
    # Therefore, the calculated raw values should be two events of 250!
    assert len(result["raw_data"]["total_charge"]) == 2
    assert result["raw_data"]["total_charge"][0] == 250
    assert result["raw_data"]["total_charge"][1] == 250
    
    # Verify the time calculation: (2.000.000 - 1.000.000) us = 1.0 seconds
    assert result["acquisition_time_sec"] == 1.0


def test_should_return_empty_dict_if_root_file_invalid():
    """
    Tests that if the ROOT file cannot be opened (e.g., due to an error),
    the function should return an empty dictionary.
    """
    # GIVEN: We simulate an error when trying to open the ROOT file
    with patch("Scripts.analysis.event_processing.uproot.open", side_effect=Exception("File not found")):
        # WHEN: We call the function
        result = collect_all_histograms("non_existent_file.root")

        # THEN: The result should be an empty dictionary
        assert result == {}

@patch("Scripts.analysis.event_processing.apply_subtraction_and_plot")
@patch("Scripts.analysis.event_processing.uproot.open")
def test_should_trigger_background_subtraction_if_bkg_data_provided(mock_uproot, mock_apply_subtraction):
    """
    Tests that if a background data dictionary is provided to collect_all_histograms,
    the function should call apply_subtraction_and_plot to perform the subtraction.
    """
    # GIVEN: Fake data for the ROOT file
    fake_data = {
        "trigTime": np.array([1000000, 2000000]),
        "channelID": np.array([[0, 1], [5]], dtype=object),
        "channelDataLG": np.array([[200, 150], [300]], dtype=object) 
    }
    mock_tree = MagicMock()
    mock_tree.arrays.return_value = fake_data
    mock_uproot.return_value.__enter__.return_value = MagicMock(__getitem__=MagicMock(return_value=mock_tree))
    
    # A fake dictionary has to be returned 
    mock_apply_subtraction.return_value = {"histograms": {}, "scatter_data": {}}
    
    # Create fake background data
    fake_bkg_data = {"sono_un": "background"}

    # WHEN: We call the function passing to it fake background data
    collect_all_histograms("fake.root", bkg_data=fake_bkg_data)

    # THEN: We check that apply_subtraction_and_plot was called once with the correct arguments
    mock_apply_subtraction.assert_called_once()