"""
io_manager.py

Centralized path management and data loading for Isolpharm.
Includes functions to resolve paths for Data_converted, Results, and background loading.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any

# BASE PATH MANAGEMENT
def get_project_root() -> str:
    """Returns the root directory of the project (Isolpharm)."""
    # Go up 2 levels from Scripts/analysis/io_manager.py -> Isolpharm
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_converted_dir(date_str: str) -> str:
    """Returns the path to the converted data folder for a specific date."""
    root = get_project_root()
    return os.path.join(root, "Data", "Data_converted", date_str)

def get_run_output_dir(date_str: str, run_name: str) -> str:
    """Returns (and creates) the output path for a Run's results."""
    root = get_project_root()
    out_dir = os.path.join(root, "Results", date_str, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

#DATA LOADING (JSON -> NUMPY)
def load_json_data_with_numpy_conversion(json_path: str) -> Dict[str, Any]:
    """
    Loads a JSON file and converts its lists into NumPy arrays.
    Necessary to perform mathematical vector subtraction.
    """
    if not os.path.exists(json_path):
        return {}

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"  (IO) Warning: The file {json_path} is empty or not a valid JSON.")
            return {}
        
    # Recursive helper or specific sections
    # Here we use logic tailored to the structure of our results dictionary
    
    # AmpliDistLG (Matrix)
    if "ampliDistLG" in data and isinstance(data["ampliDistLG"], list):
        data["ampliDistLG"] = np.array(data["ampliDistLG"])

    # Raw Data (trigTime, etc.)
    if "raw_data" in data and isinstance(data["raw_data"], dict):
        for key, value in data["raw_data"].items():
            if isinstance(value, list):
                data["raw_data"][key] = np.array(value)

    # Histograms (bins, counts)
    if "histograms" in data and isinstance(data["histograms"], dict):
        for name, content in data["histograms"].items():
            # content is a [bins, counts] list or tuple
            if len(content) == 2:
                data["histograms"][name] = (np.array(content[0]), np.array(content[1]))

    # Maps
    if "maps" in data and isinstance(data["maps"], dict):
        for name, matrix in data["maps"].items():
            if isinstance(matrix, list):
                data["maps"][name] = np.array(matrix)

    return data

#BACKGROUND REFERENCE MANAGEMENT
def load_background_reference() -> Optional[Dict[str, Any]]:
    """
    Loads the background_reference.json file from the Results folder.
    This is the function called by run_pipeline.py.
    """
    root = get_project_root()
    path = os.path.join(root, "Results", "background_reference.json")
    
    if os.path.exists(path):
        try:
            print(f"   (IO) Loading background from: {path}")
            return load_json_data_with_numpy_conversion(path)
        except Exception as e:
            print(f"   (IO) Error reading background JSON: {e}")
            return None
    else:
        print(f"   (IO) No background file found at: {path}")
        return None