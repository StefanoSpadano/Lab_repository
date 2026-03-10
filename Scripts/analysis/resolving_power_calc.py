import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analysis.io_manager import get_project_root

BIN_SIZE_MM = 1.5  # Abbassiamo il binning a 1 mm per avere più punti per il fit doppio

def double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2, c):
    """Somma di due Gaussiane spaziali + fondo costante"""
    g1 = amp1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    g2 = amp2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    return g1 + g2 + c

def load_scatter_data(run_path):
    json_path = os.path.join(run_path, "histograms", "scatter_data.json")
    if not os.path.exists(json_path):
        json_path_alt = os.path.join(run_path, f"{os.path.basename(run_path)}_results.json")
        if os.path.exists(json_path_alt):
            json_path = json_path_alt
        else:
            return None, None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if "xyclus_x" in data:
            return np.array(data["xyclus_x"]), np.array(data["xyclus_y"])
        elif "scatter_data" in data:
            return np.array(data["scatter_data"]["xyclus_x"]), np.array(data["scatter_data"]["xyclus_y"])
    except Exception as e:
        print(f"❌ Errore lettura JSON: {e}")
    return None, None

def fit_double_profile(axis_vals, profile_counts, label, ax):
    """
    Tenta un fit a DOPPIA GAUSSIANA per misurare la distanza tra due sorgenti.
    Restituisce la distanza in mm (o None se fallisce o trova un solo picco).
    """
    ax.step(axis_vals, profile_counts, where='mid', color='gray', alpha=0.5, label='Dati')
    ax.grid(alpha=0.3)
    ax.scatter(axis_vals, profile_counts, color='black', s=15, zorder=3)

    if len(axis_vals) < 6:
        return None, None, None

    # Troviamo i due picchi principali. 
    # Semplificazione: dividiamo l'asse in due metà e cerchiamo il massimo in ciascuna.
    mid_idx = len(axis_vals) // 2
    idx1 = np.argmax(profile_counts[:mid_idx])
    idx2 = mid_idx + np.argmax(profile_counts[mid_idx:])
    
    mu1_guess = axis_vals[idx1]
    mu2_guess = axis_vals[idx2]
    amp1_guess = profile_counts[idx1]
    amp2_guess = profile_counts[idx2]
    sigma_guess = 2.0  # Guess iniziale ragionevole per la singola fiala (circa la PSF nota)

    # Se i due massimi sono troppo vicini o uno è rumore, il fit doppio fallirà o non ha senso
    if abs(mu1_guess - mu2_guess) < 2.0 or amp2_guess < (0.2 * amp1_guess):
        ax.set_title(f"{label}: Single Peak Dominant", color='black')
        return None, None, None

    p0 = [amp1_guess, mu1_guess, sigma_guess, amp2_guess, mu2_guess, sigma_guess, 0]
    
    # Limiti: costringiamo i centri a non accavallarsi e le sigma a non esplodere
    bounds_low = [0, -15, 0.5, 0, -15, 0.5, 0]
    bounds_high = [np.inf, 15, 10, np.inf, 15, 10, np.inf]

    try:
        popt, _ = curve_fit(double_gaussian, axis_vals, profile_counts, p0=p0, bounds=(bounds_low, bounds_high), maxfev=100000)
    except Exception as e:
        ax.set_title(f"{label}: Double Fit Failed", color='red')
        return None, None, None

    amp1, mu1, sigma1, amp2, mu2, sigma2, c = popt
    
    # Calcolo distanza
    dist_mm = abs(mu2 - mu1)
    
    # Disegniamo il fit totale
    x_plot = np.linspace(axis_vals[0], axis_vals[-1], 200)
    ax.plot(x_plot, double_gaussian(x_plot, *popt), 'r-', linewidth=2, label='Double Fit', zorder=4)
    
    # Disegniamo le due singole gaussiane sottostanti tratteggiate
    g1_plot = amp1 * np.exp(-0.5 * ((x_plot - mu1) / sigma1)**2) + c
    g2_plot = amp2 * np.exp(-0.5 * ((x_plot - mu2) / sigma2)**2) + c
    ax.plot(x_plot, g1_plot, 'b--', alpha=0.6)
    ax.plot(x_plot, g2_plot, 'g--', alpha=0.6)
    
    ax.legend(fontsize='small')
    ax.set_title(f"{label}: Distance = {dist_mm:.1f} mm", color='blue', fontweight='bold')
    
    return dist_mm, mu1, mu2

def main():
    if len(sys.argv) < 3:
        print("Usage: python calc_resolving_power.py <date> <run_name>")
        return

    date_str = sys.argv[1]
    run_name = sys.argv[2]
    
    root_dir = get_project_root()
    run_path = os.path.join(root_dir, "Results", date_str, run_name)
    out_dir = os.path.join(run_path, "psf_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n📏 Calcolo Potere Risolutivo Multi-Sorgente su {run_name}...")

    x_mm, y_mm = load_scatter_data(run_path)
    if x_mm is None or len(x_mm) == 0:
        print("❌ Nessun dato spaziale trovato.")
        return

    # Usiamo un binning più fine per il fit doppio e proiettiamo su TUTTO l'asse
    bins_edge = np.arange(-13.0, 14.0, BIN_SIZE_MM)
    
    # Invece di fare lo "slicing" passante per il picco, facciamo la proiezione totale 
    # (marginale) per catturare entrambe le sorgenti
    heatmap, xedges, yedges = np.histogram2d(x_mm, y_mm, bins=bins_edge)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    
    profile_x = np.sum(heatmap, axis=1) # Somma lungo Y
    profile_y = np.sum(heatmap, axis=0) # Somma lungo X

    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 2)

    ax_map = fig.add_subplot(gs[0, 0])
    im = ax_map.imshow(heatmap.T, origin='lower', extent=[-13, 13, -13, 13], cmap='inferno')
    ax_map.set_title("2D Projection (Multi-Source)")
    ax_map.set_xlabel("X [mm]")
    ax_map.set_ylabel("Y [mm]")
    plt.colorbar(im, ax=ax_map)

    # Profilo X
    ax_x = fig.add_subplot(gs[1, 0])
    dist_x, mux1, mux2 = fit_double_profile(x_centers, profile_x, "Proiezione X", ax_x)

    # Profilo Y
    ax_y = fig.add_subplot(gs[0, 1])
    dist_y, muy1, muy2 = fit_double_profile(y_centers, profile_y, "Proiezione Y", ax_y)

    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis('off')
    
    info_text = f"MULTI-SOURCE RESOLUTION\n"
    info_text += f"Run: {run_name}\n"
    info_text += f"Binning: {BIN_SIZE_MM} mm\n\n"
    
    if dist_x:
        info_text += f"X Axis Resolved!\n"
        info_text += f"Distance: {dist_x:.2f} mm\n"
        info_text += f"Pos: {mux1:.1f}, {mux2:.1f}\n\n"
    if dist_y:
        info_text += f"Y Axis Resolved!\n"
        info_text += f"Distance: {dist_y:.2f} mm\n"
        info_text += f"Pos: {muy1:.1f}, {muy2:.1f}\n"

    ax_info.text(0.1, 0.5, info_text, fontsize=12, family='monospace', verticalalignment='center')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "Resolving_Power_Analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"✅ Analisi Multi-Sorgente salvata in: {out_path}")

if __name__ == "__main__":
    main()