"""
fitting.py

Modulo per eseguire fit gaussiani 1D e 2D.
Restituisce risultati tramite dataclass FitResult.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import curve_fit

# =============================
# Dataclass risultato fit
# =============================
@dataclass
class FitResult:
    params: np.ndarray   # parametri ottimali
    errors: np.ndarray   # errori (deviazioni standard) sui parametri
    cov: np.ndarray      # matrice di covarianza

# =============================
# Modello Gaussiano 1D
# =============================
def gaussian_model(x, A, mu, sigma, offset):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset

# =============================
# Stima iniziale per fit 1D
# =============================
def initial_guess_1d(x: np.ndarray, y: np.ndarray):
    A = np.max(y)
    mu = x[np.argmax(y)]
    sigma = (x.max() - x.min()) / 6 if x.max() != x.min() else 1.0
    offset = np.min(y)
    return [A, mu, sigma, offset]

# =============================
# Fit Gaussiano 1D
# =============================
def gaussian_fit(x: np.ndarray, y: np.ndarray, p0: Optional[list] = None) -> Optional[FitResult]:
    if len(x) == 0 or len(y) == 0:
        return None

    if p0 is None:
        p0 = initial_guess_1d(x, y)

    try:
        popt, pcov = curve_fit(gaussian_model, x, y, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        return FitResult(popt, perr, pcov)
    except Exception:
        return None

# =============================
# Modello Gaussiano 2D
# =============================
def gaussian2d_model(coord, A, x0, y0, sx, sy, offset):
    x, y = coord
    g = A * np.exp(-(((x - x0)**2) / (2 * sx**2) + ((y - y0)**2) / (2 * sy**2)))
    return (g + offset).ravel()

# =============================
# Stima iniziale per fit 2D
# =============================
def initial_guess_2d(z: np.ndarray):
    A = np.max(z)
    offset = np.min(z)

    # Trova il pixel massimo come centro iniziale
    idx = np.unravel_index(np.argmax(z), z.shape)
    x0, y0 = idx

    # Sigma iniziali
    sx = sy = 1.0

    return [A, x0, y0, sx, sy, offset]

# =============================
# Fit Gaussiano 2D
# =============================
def gaussian2d_fit(xgrid: np.ndarray, ygrid: np.ndarray, zdata: np.ndarray, p0: Optional[list] = None) -> Optional[FitResult]:
    if zdata.size == 0:
        return None

    if p0 is None:
        p0 = initial_guess_2d(zdata)

    try:
        coords = np.vstack((xgrid.ravel(), ygrid.ravel()))
        popt, pcov = curve_fit(gaussian2d_model, coords, zdata.ravel(), p0=p0)
        perr = np.sqrt(np.diag(pcov))
        return FitResult(popt, perr, pcov)
    except Exception:
        return None
