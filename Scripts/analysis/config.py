"""
config.py

Carica il file config.yaml e rende disponibile un oggetto 'cfg'
contenente tutti i parametri di configurazione.
"""

import os
import yaml

class Config:
    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, value)

# ------------------------------------------------------------
# Funzione: caricare config.yaml
# ------------------------------------------------------------
def load_config():
    # Cerca config.yaml nella stessa cartella di questo file
    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, 'config.yaml')

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"File di configurazione non trovato: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return Config(data)


# Istanza globale utilizzabile da ogni modulo
cfg = load_config()
