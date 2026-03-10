"""
io_manager.py

Gestione centralizzata dei percorsi e del caricamento dati per Isolpharm.
Include funzioni per risolvere i path di Data_converted, Results e caricare il background.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any

# ------------------------------------------------------------
# 1. GESTIONE PERCORSI BASE
# ------------------------------------------------------------
def get_project_root() -> str:
    """Restituisce la directory root del progetto (Isolpharm)."""
    # Risale di 2 livelli da Scripts/analysis/io_manager.py -> Isolpharm
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_converted_dir(date_str: str) -> str:
    """Restituisce il percorso della cartella dati convertiti per una data specifica."""
    root = get_project_root()
    return os.path.join(root, "Data", "Data_converted", date_str)

def get_run_output_dir(date_str: str, run_name: str) -> str:
    """Restituisce (e crea) il percorso di output per i risultati di una Run."""
    root = get_project_root()
    out_dir = os.path.join(root, "Results", date_str, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ------------------------------------------------------------
# 2. CARICAMENTO DATI (JSON -> NUMPY)
# ------------------------------------------------------------
def load_json_data_with_numpy_conversion(json_path: str) -> Dict[str, Any]:
    """
    Carica un JSON e converte le liste in array NumPy.
    Necessario per poter sottrarre vettori matematicamente.
    """
    if not os.path.exists(json_path):
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Helper ricorsivo o sezioni specifiche
    # Qui usiamo la logica specifica per la struttura del nostro dizionario risultati
    
    # 1. AmpliDistLG (Matrice)
    if "ampliDistLG" in data and isinstance(data["ampliDistLG"], list):
        data["ampliDistLG"] = np.array(data["ampliDistLG"])

    # 2. Raw Data (trigTime, etc.)
    if "raw_data" in data and isinstance(data["raw_data"], dict):
        for key, value in data["raw_data"].items():
            if isinstance(value, list):
                data["raw_data"][key] = np.array(value)

    # 3. Histograms (bins, counts)
    if "histograms" in data and isinstance(data["histograms"], dict):
        for name, content in data["histograms"].items():
            # content è una lista [bins, counts] o tupla
            if len(content) == 2:
                data["histograms"][name] = (np.array(content[0]), np.array(content[1]))

    # 4. Maps
    if "maps" in data and isinstance(data["maps"], dict):
        for name, matrix in data["maps"].items():
            if isinstance(matrix, list):
                data["maps"][name] = np.array(matrix)

    return data

# ------------------------------------------------------------
# 3. GESTIONE BACKGROUND REFERENCE
# ------------------------------------------------------------
def load_background_reference() -> Optional[Dict[str, Any]]:
    """
    Carica il file background_reference.json dalla cartella Results.
    Questa è la funzione chiamata da run_pipeline.py.
    """
    root = get_project_root()
    path = os.path.join(root, "Results", "background_reference.json")
    
    if os.path.exists(path):
        try:
            print(f"   (IO) Caricamento background da: {path}")
            return load_json_data_with_numpy_conversion(path)
        except Exception as e:
            print(f"   (IO) Errore lettura background JSON: {e}")
            return None
    else:
        print(f"   (IO) Nessun file di background trovato in: {path}")
        return None