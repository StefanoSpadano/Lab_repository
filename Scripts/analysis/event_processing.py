"""
event_processing.py

Modulo per la processazione degli eventi FERS a partire dai file ROOT.
Responsabilità:
- Leggere il TTree "fersTree" usando uproot
- Processare ogni evento (pixel max, cluster 5x5 e 7x7, baricentri, mappe 8x8)
- Aggregare i risultati in una struttura EventData
- Utilizzare i parametri di configurazione dal file config.yaml

Questo modulo sostituisce la funzione monolitica originale collect_all_histograms.

Versione iniziale: struttura chiara, testabile, con controlli robusti.
"""

import numpy as np
import uproot
from analysis.config import cfg
import logging
from analysis.fitting import gaussian_fit, gaussian2d_fit


# Abilita logging locale
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------
# Classe contenitore per i risultati dell'elaborazione
# ------------------------------------------------------------
class EventData:
    def __init__(self):
        # Istogrammi ADC per canale: lista di 64 array (4096 bin ciascuno)
        self.ampliDistLG = [np.zeros(4096, dtype=float) for _ in range(64)]

        # Liste di variabili evento per istogrammi 1D/2D
        self.total_charge = []
        self.main_pix_charge = []
        self.main_clust_charge = []

        self.npix5 = []
        self.npix7 = []
        self.npixdiff57 = []

        self.xglob = []
        self.yglob = []
        self.xclus = []
        self.yclus = []

        # Mappe 8x8 globali e di cluster
        self.pixmapglob = np.zeros((8, 8), dtype=float)
        self.pixmapclus = np.zeros((8, 8), dtype=float)


# ------------------------------------------------------------
# Costruzione istogrammi e fit dei risultati raccolti
# ------------------------------------------------------------
def compute_fits_from_eventdata(out: EventData):
    """
    Costruisce gli istogrammi 1D/2D dai dati della run
    e calcola i fit gaussiani.
    Restituisce un dizionario 'fits' e uno 'histograms'.
    """

    # --- Istogramma della carica totale ---
    charge_hist, charge_bins = np.histogram(out.total_charge, bins=50)
    charge_centers = 0.5 * (charge_bins[:-1] + charge_bins[1:])
    charge_fit = gaussian_fit(charge_centers, charge_hist)

    # --- Istogramma xglob ---
    x_hist, x_bins = np.histogram(out.xglob, bins=50)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    x_fit = gaussian_fit(x_centers, x_hist)

    # --- Istogramma yglob ---
    y_hist, y_bins = np.histogram(out.yglob, bins=50)
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    y_fit = gaussian_fit(y_centers, y_hist)

    # --- Fit 2D della mappa cluster (8x8) ---
    xgrid, ygrid = np.meshgrid(np.arange(8), np.arange(8))
    fit2d = gaussian2d_fit(xgrid, ygrid, out.pixmapclus)

    return {
        "charge": charge_fit,
        "xglob": x_fit,
        "yglob": y_fit,
        "spot2d": fit2d
    }, {
        "charge": (charge_centers, charge_hist),
        "xglob": (x_centers, x_hist),
        "yglob": (y_centers, y_hist)
    }


# ------------------------------------------------------------
# Funzione di utilità: da channel ID a coordinate (0..7, 0..7)
# ------------------------------------------------------------
def channel_id_to_xy(cid):
    """
    Conversione semplificata ID → coordinate 8x8.
    Nota: nel vecchio script usavi liste chID2mapX e chID2mapY.
    Qui assumiamo una mappa standard row-major.
    Se necessario la sostituiremo con la mappa reale.
    """
    x = cid % 8
    y = cid // 8
    return x, y


# ------------------------------------------------------------
# Processamento di UN SOLO evento
# ------------------------------------------------------------
def process_single_event(ids, vals):
    """
    Processa un singolo evento FERS.
    Calcola:
    - pixel massimo
    - carica totale e cariche cluster
    - numero pixel nei cluster 5x5 e 7x7
    - baricentri globali e cluster
    - mappe 8x8 locali

    Ritorna un dict con i contributi dell'evento.
    """

    thr = cfg.threshold
    ped = cfg.pedestal
    pix = cfg.pixel_size

    # Se evento vuoto
    if len(vals) == 0:
        return None

    # Pixel max
    max_i = np.argmax(vals)
    max_cid = ids[max_i]
    max_v = vals[max_i]

    # Evento saturo
    if max_v >= 4000:
        return None

    # Coordinate max pixel
    max_x, max_y = channel_id_to_xy(max_cid)

    # Variabili cumulative
    total_charge = 0.0
    main_pix_charge = 0.0
    main_clust_charge = 0.0
    np5 = 0
    np7 = 0

    posgx = 0.0
    posgy = 0.0
    chglob = 0.0

    poscx = 0.0
    poscy = 0.0
    chclus = 0.0

    # Mappe locali
    pixmapglob_ev = np.zeros((8, 8))
    pixmapclus_ev = np.zeros((8, 8))

    # Loop sui canali dell'evento
    for cid, v in zip(ids, vals):
        if v < thr:
            continue

        adc = v - ped
        if adc <= 0:
            continue

        # Aggiorna totale
        total_charge += adc

        # Coordinate pixel
        x, y = channel_id_to_xy(cid)
        pixmapglob_ev[y, x] += v / 4000.0

        # Baricentro globale
        chglob += adc
        posgx += adc * x
        posgy += adc * y

        # Pixel massimo
        if cid == max_cid:
            main_pix_charge = adc

        # Cluster 5x5
        if abs(x - max_x) <= 2 and abs(y - max_y) <= 2:
            np5 += 1
            main_clust_charge += adc
            pixmapclus_ev[y, x] += v / 4000.0
            chclus += adc
            poscx += adc * x
            poscy += adc * y

        # Cluster 7x7
        if abs(x - max_x) <= 3 and abs(y - max_y) <= 3:
            np7 += 1

    # Controllo eventi senza carica
    if chglob == 0:
        return None

    # Coordinate baricentro globale
    xg = ((posgx / chglob) - 3.5) * pix
    yg = ((posgy / chglob) - 3.5) * pix

    # Coordinate cluster
    if chclus > 0:
        xc = ((poscx / chclus) - 3.5) * pix
        yc = ((poscy / chclus) - 3.5) * pix
    else:
        xc = None
        yc = None

    return {
        "total_charge": total_charge,
        "main_pix_charge": main_pix_charge,
        "main_clust_charge": main_clust_charge,
        "np5": np5,
        "np7": np7,
        "npdiff": np7 - np5,
        "xg": xg,
        "yg": yg,
        "xc": xc,
        "yc": yc,
        "pixmapglob": pixmapglob_ev,
        "pixmapclus": pixmapclus_ev,
        "max_cid": max_cid,
        "vals": vals,
    }


# ------------------------------------------------------------
# Funzione principale: processa TUTTI gli eventi del file ROOT
# ------------------------------------------------------------
def collect_all_histograms(root_path):
    """
    Legge il file ROOT e aggrega gli eventi in un oggetto EventData.
    """

    logger.info(f"Apertura file ROOT: {root_path}")
    out = EventData()

    with uproot.open(root_path) as f:
        tree = f["fersTree"]
        arrays = tree.arrays(["channelID", "channelDataLG"], library="np")

    channel_ids = arrays["channelID"]
    channel_vals = arrays["channelDataLG"]

    n_events = len(channel_vals)
    logger.info(f"Trovati {n_events} eventi.")

    # Loop eventi
    for ev in range(n_events):
        ids = channel_ids[ev]
        vals = channel_vals[ev]

        res = process_single_event(ids, vals)
        if res is None:
            continue

        # Aggiorna EventData
        out.total_charge.append(res["total_charge"])
        out.main_pix_charge.append(res["main_pix_charge"])
        out.main_clust_charge.append(res["main_clust_charge"])

        out.npix5.append(res["np5"])
        out.npix7.append(res["np7"])
        out.npixdiff57.append(res["npdiff"])

        out.xglob.append(res["xg"])
        out.yglob.append(res["yg"])

        if res["xc"] is not None:
            out.xclus.append(res["xc"])
            out.yclus.append(res["yc"])

        out.pixmapglob += res["pixmapglob"]
        out.pixmapclus += res["pixmapclus"]

        # Aggiorna istogrammi ADC per pixel
        vals = res["vals"]
        for cid, v in zip(ids, vals):
            if 0 <= v < 4096:
                out.ampliDistLG[cid][int(v)] += 1

    # Fine processamento eventi
    logger.info("Processamento completato.")

    # ---- Calcolo istogrammi e fit ----
    fits, histograms = compute_fits_from_eventdata(out)

    return {
        "raw": out,
        "fits": fits,
        "histograms": histograms,
        "maps": {
            "pixmapglob": out.pixmapglob,
            "pixmapclus": out.pixmapclus
        }
    }


