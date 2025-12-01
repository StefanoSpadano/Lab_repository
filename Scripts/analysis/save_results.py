import os
import json
from typing import Dict
import numpy as np
from .fitting import FitResult


# ----------------------------------------
# Create directory if missing
# ----------------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------------------
# Convert FitResult → JSON serializable dict
# ----------------------------------------
def serialize_fit_result(fit: FitResult) -> Dict:
    if fit is None:
        return {}

    return {
        "params": fit.params.tolist(),   # numpy → list
        "errors": fit.errors.tolist() if fit.errors is not None else None,
        "covariance": fit.cov.tolist() if fit.cov is not None else None
    }


# ----------------------------------------
# Save FitResult to JSON
# ----------------------------------------
def save_fit_result(fit: FitResult, out_path: str):
    ensure_dir(os.path.dirname(out_path))

    out_dict = serialize_fit_result(fit)

    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=4)

    print(f"[OK] Salvato fit JSON → {out_path}")


# ----------------------------------------
# Save histogram data
# ----------------------------------------
def save_histogram_data(bin_centers, hist_values, out_path: str):
    ensure_dir(os.path.dirname(out_path))

    out = {
        "bin_centers": bin_centers.tolist(),
        "hist_values": hist_values.tolist()
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"[OK] Salvato istogramma → {out_path}")


# ----------------------------------------
# Save map/matrix
# ----------------------------------------
def save_map_data(matrix: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path))

    out = {"matrix": matrix.tolist()}

    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"[OK] Salvata mappa → {out_path}")


# ----------------------------------------
# Save figure (PNG)
# ----------------------------------------
def save_figure(fig, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    print(f"[OK] Salvato plot → {out_path}")

# ----------------------------------------
# Save total-charge spectrum (JSON + PNG)
# ----------------------------------------
def save_total_charge_spectrum(res, base_path: str):
    """
    Crea e salva lo spettro di total_charge usando binning automatico.
    Salva:
        - spectrum/total_charge_spectrum.json
        - spectrum/total_charge_spectrum.png
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # estrai vettore total_charge
    total = np.array(res["raw"].total_charge, dtype=float)

    # istogramma automatico numpy
    hist_vals, bin_edges = np.histogram(total, bins="auto")

    # directory output
    spectrum_dir = os.path.join(base_path, "spectrum")
    ensure_dir(spectrum_dir)

    # 1) salva JSON
    out_json = {
        "bin_edges": bin_edges.tolist(),
        "hist_values": hist_vals.tolist(),
        "n_events": len(total),
        "min_charge": float(np.min(total)),
        "max_charge": float(np.max(total))
    }

    with open(os.path.join(spectrum_dir, "total_charge_spectrum.json"), "w") as f:
        json.dump(out_json, f, indent=4)

    print(f"[OK] Salvato JSON spettro carica totale → {spectrum_dir}/total_charge_spectrum.json")

    # 2) salva figura PNG
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(total, bins=bin_edges)
    ax.set_title("Total Charge Spectrum")
    ax.set_xlabel("Total charge (ADC)")
    ax.set_ylabel("Counts")
    fig.tight_layout()
    fig.savefig(os.path.join(spectrum_dir, "total_charge_spectrum.png"), dpi=150)
    plt.close(fig)

    print(f"[OK] Salvato PNG spettro carica totale → {spectrum_dir}/total_charge_spectrum.png")


# ----------------------------------------
# Main function: save ALL results
# ----------------------------------------
def save_all_results(res, figs: dict, date_str: str, run_number: int):
    """
    res: output di collect_all_histograms
    figs: dict con nome → figure matplotlib
    date_str: "2025-11-06"
    run_number: es. 1
    """

    base = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "..",
        "Results",
        date_str,
        f"Run{run_number}"
    ))

    ensure_dir(base)

    # --------------------------
    # 1) FIT RESULTS
    # --------------------------
    fits_path = os.path.join(base, "fits")
    ensure_dir(fits_path)

    for name, fit in res["fits"].items():
        save_fit_result(fit, os.path.join(fits_path, f"{name}.json"))

    # --------------------------
    # 2) HISTOGRAMS
    # --------------------------
    hist_path = os.path.join(base, "histograms")
    ensure_dir(hist_path)

    for name, (bins, values) in res["histograms"].items():
        save_histogram_data(bins, values, os.path.join(hist_path, f"{name}.json"))

    # --------------------------
    # 3) MAPS
    # --------------------------
    maps_path = os.path.join(base, "maps")
    ensure_dir(maps_path)

    save_map_data(res["maps"]["pixmapglob"], os.path.join(maps_path, "pixmapglob.json"))
    save_map_data(res["maps"]["pixmapclus"], os.path.join(maps_path, "pixmapclus.json"))

    # --------------------------
    # 4) PLOTS (PNG)
    # --------------------------
    fig_path = os.path.join(base, "plots")
    ensure_dir(fig_path)

    for name, fig in figs.items():
        save_figure(fig, os.path.join(fig_path, f"{name}.png"))
    
    # --------------------------
    # 5) TOTAL CHARGE SPECTRUM
    # --------------------------
    save_total_charge_spectrum(res, base)

    print("\n================================")
    print("   ✔️ TUTTI I RISULTATI SALVATI ")
    print("================================\n")
