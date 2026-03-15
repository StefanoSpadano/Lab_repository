import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)

from Scripts.analysis.save_results import convert_numpy

def test_should_convert_numpy_types_to_standard_python_types():
    """
    Tests that the utility function convert_numpy correctly converts NumPy types
    (integers, floats, arrays) into standard Python types (int, float, list) for JSON serialization.
    """
    # GIVEN: Some variables of NumPy type and a standard Python variable
    np_int = np.int64(42)
    np_float = np.float32(3.14)
    np_array = np.array([1, 2, 3])
    standard_string = "non toccarmi"

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
    assert res_str == "non toccarmi"