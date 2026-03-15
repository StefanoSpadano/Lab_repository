import os
import sys
import numpy as np
import json
from unittest.mock import patch

current_dir = os.path.dirname(os.path.abspath(__file__))
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)

from Scripts.analysis.save_results import convert_numpy, save_all_results

def test_should_convert_numpy_types_to_standard_python_types():
    """
    Tests that the utility function convert_numpy correctly converts NumPy types
    (integers, floats, arrays) into standard Python types (int, float, list) for JSON serialization.
    """
    # GIVEN: Some variables of NumPy type and a standard Python variable
    np_int = np.int64(42)
    np_float = np.float32(3.14)
    np_array = np.array([1, 2, 3])
    standard_string = "This is a standard string, not a NumPy type."

    # WHEN: These variables are passed to the conversion function
    res_int = convert_numpy(np_int)
    res_float = convert_numpy(np_float)
    res_array = convert_numpy(np_array)
    res_str = convert_numpy(standard_string)

    # THEN: We check that the types have been converted correctly
    assert isinstance(res_int, int)
    assert res_int == 42
    
    assert isinstance(res_float, float)
    
    assert isinstance(res_array, list)
    assert res_array == [1, 2, 3]
    
    # The string (not being NumPy) should remain unchanged
    assert isinstance(res_str, str)
    assert res_str == "This is a standard string, not a NumPy type."

def test_should_convert_2d_numpy_array_to_list_of_lists():
    """
    Tests that a 2D NumPy array is correctly converted into a list of lists (nested lists).
    This is important for JSON serialization of matrices (like ampliDistLG).
    """
    # GIVEN: A 2D NumPy array
    np_2d_array = np.array([[1, 2], [3, 4]])

    # WHEN: The array is passed to the conversion function
    res = convert_numpy(np_2d_array)

    # THEN: We check that the result is a list of lists with the same values
    assert isinstance(res, list)
    assert res == [[1, 2], [3, 4]]

@patch("Scripts.analysis.save_results.get_run_output_dir")
@patch("Scripts.analysis.save_results.plt.savefig")
@patch("Scripts.analysis.save_results.plt.figure")
@patch("Scripts.analysis.save_results.plt.close")
def test_should_save_results_and_create_legacy_plots(mock_close, mock_figure, mock_savefig, mock_get_dir, tmp_path):
    """
    Tests that the save_all_results function correctly saves the results in JSON format and creates the expected plot files in the correct folders. 
    """
    # GIVEN: A temporary directory initialized in the usual method (tmp_path)
    mock_get_dir.return_value = str(tmp_path)
    
    # We prepare a dictionary of fake and minimal results
    fake_results = {
        "histograms": {
            # Bins has N elements, counts has N-1 elements instead. This is the standard format for histograms (bin edges and counts).
            "total_charge": (np.array([0, 1, 2]), np.array([100, 200]))
        },
        "scatter_data": {
            "xyglob_x": np.array([1.0, 2.0]),
            "xyglob_y": np.array([1.5, 2.5]),
            # We add data to force the saving of the scatter JSON
            "extra_data": np.int64(10) 
        }
    }

    # WHEN: The save_all_results function is called
    save_all_results(fake_results, "01_01_2026", "Run0")

    # THEN: 
    # We start by verifying if the folder structure has been created correctly
    hist_dir = tmp_path / "histograms"
    plots_dir = tmp_path / "plots"
    maps_dir = tmp_path / "maps"
    
    assert hist_dir.exists()
    assert plots_dir.exists()
    assert maps_dir.exists()
    
    # Verify that the JSON files have been written correctly
    charge_json_path = hist_dir / "total_charge.json"
    assert charge_json_path.exists()
    with open(charge_json_path, "r") as f:
        data = json.load(f)
        assert data["bins"] == [0, 1, 2]
        assert data["counts"] == [100, 200]
        
    # Check scatter_data.json
    scatter_json_path = hist_dir / "scatter_data.json"
    assert scatter_json_path.exists()
    with open(scatter_json_path, "r") as f:
        scatter_data = json.load(f)
        assert scatter_data["xyglob_x"] == [1.0, 2.0]
        assert scatter_data["extra_data"] == 10  # int64 has to be converted to int for JSON
        
    # Verify that matplotlib has been called to save the images
    # (One call for the histogram, one for the 2D plot xyglob_sub.png)
    assert mock_savefig.call_count >= 2


@patch("Scripts.analysis.save_results.get_run_output_dir")
def test_should_handle_empty_results_gracefully(mock_get_dir, tmp_path):
    """
    Tests that the save_all_results function can handle an empty results dictionary 
    without crashing and still creates the necessary folders.
    """
    # We tell the function to create the folders inside our fake temporary folder
    mock_get_dir.return_value = str(tmp_path)
    
    # GIVEN: An empty results dictionary
    empty_results = {}

    # WHEN: The save_all_results function is called with empty results
    save_all_results(empty_results, "01_01_2026", "RunEmpty")

    # THEN: We check that the folders are created and no exceptions are raised
    hist_dir = tmp_path / "histograms"
    plots_dir = tmp_path / "plots"
    maps_dir = tmp_path / "maps"
    
    assert hist_dir.exists()
    assert plots_dir.exists()
    assert maps_dir.exists()


# The patch order matters: the patch closest to the function being tested becomes the first argument (mock_save_fig)
@patch("Scripts.analysis.save_results.get_run_output_dir")
@patch("Scripts.analysis.save_results.plt.savefig")
def test_should_continue_execution_if_scatter_json_fails(mock_save_fig, mock_get_dir, tmp_path):
    """
    Tests that if there is an error in saving the scatter_data JSON (e.g., due to non-serializable data), 
    the function should catch the exception, print an error message, and continue execution without crashing.
    """
    # We tell the function to create the folders inside our fake temporary folder
    mock_get_dir.return_value = str(tmp_path)
    
    # GIVEN: A results dictionary with non-serializable data in scatter_data
    non_serializable_results = {
        "scatter_data": {
            "xyglob_x": np.array([1.0, 2.0]),
            "xyglob_y": np.array([1.5, 2.5]),
            "non_serializable": set([1, 2, 3])  # Sets are not JSON serializable
        }
    }

    # WHEN: The save_all_results function is called with non-serializable data
    save_all_results(non_serializable_results, "01_01_2026", "RunNonSerializable")

    # THEN: We check that folders are created
    hist_dir = tmp_path / "histograms"
    plots_dir = tmp_path / "plots"
    maps_dir = tmp_path / "maps"
    
    assert hist_dir.exists()
    assert plots_dir.exists()
    assert maps_dir.exists()
    
    # The JSON file is created by open(), but since json.dump fails, it will be empty (0 bytes). 
    scatter_json_path = hist_dir / "scatter_data.json"
    assert scatter_json_path.exists()
    # So the JSON file is being created before it crashes and because of that we cannot test that it is exactly 0 bytes. 
    
    # The program should have continued and attempted to save the plots, so plt.savefig should have been called at least once (for the 2D plot)
    assert mock_save_fig.call_count > 0