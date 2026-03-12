import os
import sys
import json
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)

from Scripts.analysis.io_manager import load_json_data_with_numpy_conversion

def test_should_convert_json_lists_to_numpy_arrays(tmp_path):
    """
    Tests that the loaded JSON dictionary "sees" the lists
    correctly converted in a NumPy array according to the function's logic. 
    """
    # GIVEN: A fake JSON file with the exact structure we expect, but filled with normal lists
    fake_dict = {
        "ampliDistLG": [100, 200, 300],
        "raw_data": {
            "trigTime": [10.5, 20.1, 30.8]
        },
        "histograms": {
            "energy_spectrum": [[0, 1, 2], [10, 50, 10]] # [bins, counts]
        },
        "maps": {
            "hitmap": [[0, 1], [1, 0]]
        },
        "ignored_string": "this has to remain as it is"
    }
    
    # We create the file and plug in the dictionary in JSON format
    fake_json_file = tmp_path / "fake_results.json"
    with open(fake_json_file, "w") as f:
        json.dump(fake_dict, f)

    # WHEN: The loading function is called 
    result = load_json_data_with_numpy_conversion(str(fake_json_file))

    # THEN: We check that the conversion has been successfully made with the type being now a numpy.ndarray
    assert isinstance(result["ampliDistLG"], np.ndarray)
    assert isinstance(result["raw_data"]["trigTime"], np.ndarray)
    assert isinstance(result["maps"]["hitmap"], np.ndarray)
    
    # Histogram are tuple of 2d arrays
    assert isinstance(result["histograms"]["energy_spectrum"], tuple)
    assert isinstance(result["histograms"]["energy_spectrum"][0], np.ndarray) # Bins
    assert isinstance(result["histograms"]["energy_spectrum"][1], np.ndarray) # Counts
    
    #check that the data that had to be left alone are actually left alone
    assert result["ignored_string"] == "this has to remain as it is"

def test_should_return_empty_dict_if_file_does_not_exist():
    """
    Tests that if the file does not exist, the function returns an empty dictionary.
    """
    # GIVEN: A path to a non-existent file
    non_existent_file = "this_file_does_not_exist.json"

    # WHEN: The loading function is called 
    result = load_json_data_with_numpy_conversion(non_existent_file)

    # THEN: We check that the result is an empty dictionary
    assert isinstance(result, dict)
    assert len(result) == 0

def test_should_handle_empty_json_file(tmp_path):
    """
    Tests that if the file is empty, the function returns an empty dictionary.
    """
    # GIVEN: An empty JSON file
    empty_json_file = tmp_path / "empty.json"
    with open(empty_json_file, "w") as f:
        f.write("")  # Write an empty string to create an empty file

    # WHEN: The loading function is called 
    result = load_json_data_with_numpy_conversion(str(empty_json_file))

    # THEN: We check that the result is an empty dictionary
    assert isinstance(result, dict)
    assert len(result) == 0

def test_should_handle_non_json_content(tmp_path):
    """
    Tests that if the file content is not a valid JSON, the function returns an empty dictionary.
    """
    # GIVEN: A fake file with non-JSON content
    fake_file = tmp_path / "not_a_json.txt"
    with open(fake_file, "w") as f:
        f.write("This is not a JSON content")

    # WHEN: The loading function is called 
    result = load_json_data_with_numpy_conversion(str(fake_file))

    # THEN: We check that the result is an empty dictionary
    assert isinstance(result, dict)
    assert len(result) == 0