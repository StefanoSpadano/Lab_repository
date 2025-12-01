"""
io_manager.py

Gestione dei file e della struttura delle cartelle del progetto Isolpharm.
Responsabilità:
- Identificare la cartella dati più recente
- Enumerare i file RunN.root
- Individuare il background
- Restituire percorsi consistenti

Questa è una prima versione estratta e pulita dal vecchio script.
"""

import os
from datetime import datetime
from typing import Dict, Optional

# ------------------------------------------------------------
# Funzione: ottenere root del progetto
# ------------------------------------------------------------
def get_project_root() -> str:
    """Restituisce la directory root del progetto Isolpharm."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ------------------------------------------------------------
# Funzione: trovare la cartella dati più recente (YYYY-MM-DD)
# ------------------------------------------------------------
def get_latest_data_folder(base_dir: str) -> Optional[str]:
    """
    Cerca nel percorso base_dir tutte le directory con nome YYYY-MM-DD
    e restituisce quella più recente.
    """
    if not os.path.isdir(base_dir):
        return None

    candidates = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            try:
                # prova a interpretare la cartella come data
                datetime.strptime(name, "%Y-%m-%d")
                candidates.append(full)
            except ValueError:
                pass

    if not candidates:
        return None

    # ordina per data
    candidates.sort(key=lambda p: datetime.strptime(os.path.basename(p), "%Y-%m-%d"))
    return candidates[-1]


# ------------------------------------------------------------
# Funzione: enumerare file Run*.root
# ------------------------------------------------------------
def list_run_files(folder: str) -> Dict[int, str]:
    """
    Ritorna un dizionario {run_number: file_path} per tutti i file RunN.root.
    """
    runs = {}
    for fname in os.listdir(folder):
        if fname.startswith("Run") and fname.endswith(".root"):
            middle = fname[3:-5]  # estrae il numero tra 'Run' e '.root'
            if middle.isdigit():
                runs[int(middle)] = os.path.join(folder, fname)
    return dict(sorted(runs.items()))


# ------------------------------------------------------------
# Funzione: trovare background
# ------------------------------------------------------------
def find_background_file() -> Optional[str]:
    """
    Cerca il file di background nelle due posizioni standard.
    """
    root = get_project_root()
    path1 = os.path.join(root, "Data", "Data_converted", "background", "background.root")
    path2 = os.path.join(root, "Data", "Data_converted", "background.root")

    if os.path.isfile(path1):
        return path1
    if os.path.isfile(path2):
        return path2
    return None
