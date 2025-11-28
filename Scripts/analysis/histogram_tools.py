"""
histogram_tools.py

Modulo dedicato alla costruzione e manipolazione degli istogrammi.
Versione A (minimale):
- istogramma 1D
- istogramma 2D
- sottrazione background
- clipping dei valori negativi
- calcolo dei centri bin

Questo modulo non fa fit, non crea figure e non contiene logica sugli eventi.
Serve come fondazione per i moduli fitting.py e plotting.py.
"""

import numpy as np

# ------------------------------------------------------------
# Istogramma 1D
# ------------------------------------------------------------
def build_histogram(data, bins):
    """
    Costruisce un istogramma 1D.
    data: lista o array di valori
    bins: numero bin o array bordi bin
    Ritorna (hist, bin_edges)
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges


# ------------------------------------------------------------
# Istogramma 2D
# ------------------------------------------------------------
def build_histogram2d(x, y, bins):
    """
    Costruisce un istogramma 2D.
    x, y: liste o array di valori
    bins: [xbins, ybins] oppure numero bin uniforme
    Ritorna (hist2d, xedges, yedges)
    """
    hist2d, xedges, yedges = np.histogram2d(x, y, bins=bins)
    return hist2d, xedges, yedges


# ------------------------------------------------------------
# Sottrazione background
# ------------------------------------------------------------
def subtract_histograms(h, h_bkg, scale=1.0):
    """
    Sottrae h_bkg da h con un fattore di scala.
    Valori negativi vengono messi a zero.
    """
    h_sub = h - scale * h_bkg
    h_sub[h_sub < 0] = 0
    return h_sub


# ------------------------------------------------------------
# Clipping dei negativi
# ------------------------------------------------------------
def clip_negative(h):
    """
    Imposta a zero i valori negativi.
    Utile dopo la sottrazione.
    """
    h[h < 0] = 0
    return h


# ------------------------------------------------------------
# Centri bin
# ------------------------------------------------------------
def compute_bin_centers(bin_edges):
    """
    Dati i bordi dei bin, calcola i centri.
    """
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])
