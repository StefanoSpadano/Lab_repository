import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analysis.io_manager import get_project_root

BIN_SIZE_MM = 1.5  

def single_gaussian(x, amp, mu, sigma, c):
    """Gaussiana singola di ripiego per sorgenti allineate"""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + c

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
    Tenta un fit a DOPPIA GAUSSIANA. Se fallisce o i picchi sono allineati,
    esegue un fallback a una SINGOLA GAUSSIANA per garantire la visualizzazione.
    Restituisce: distanza_mm (0 se singolo picco), mu1, mu2 (None se singolo).
    """
    ax.step(axis_vals, profile_counts, where='mid', color='gray', alpha=0.5, label='Data', linewidth=1.5)
    ax.grid(alpha=0.3)
    ax.scatter(axis_vals, profile_counts, color='black', s=20, zorder=3)

    if len(axis_vals) < 6:
        return None, None, None
    
    if np.max(profile_counts) == 0:
        ax.text(0.5, 0.5, "No data (all zeros)", transform=ax.transAxes, ha='center')
        return None, None, None

    # Trova i due picchi principali dividendo a metà
    mid_idx = len(axis_vals) // 2
    idx1 = np.argmax(profile_counts[:mid_idx])
    idx2 = mid_idx + np.argmax(profile_counts[mid_idx:])
    
    mu1_guess = axis_vals[idx1]
    mu2_guess = axis_vals[idx2]
    amp1_guess = profile_counts[idx1]
    amp2_guess = profile_counts[idx2]
    sigma_guess = 2.0  

    is_single_peak = False
    
    # Condizione: i massimi sono troppo vicini o uno è trascurabile
    if abs(mu1_guess - mu2_guess) < 2.0 or amp2_guess < (0.2 * amp1_guess) or amp1_guess < (0.2 * amp2_guess):
        is_single_peak = True

    if not is_single_peak:
        p0 = [amp1_guess, mu1_guess, sigma_guess, amp2_guess, mu2_guess, sigma_guess, 0]
        bounds_low = [0, -15, 0.5, 0, -15, 0.5, 0]
        bounds_high = [np.inf, 15, 10, np.inf, 15, 10, np.inf]
        try:
            popt, _ = curve_fit(double_gaussian, axis_vals, profile_counts, p0=p0, bounds=(bounds_low, bounds_high), maxfev=100000)
            amp1, mu1, sigma1, amp2, mu2, sigma2, c = popt
            dist_mm = abs(mu2 - mu1)
            
            x_plot = np.linspace(axis_vals[0], axis_vals[-1], 200)
            ax.plot(x_plot, double_gaussian(x_plot, *popt), 'r-', linewidth=2.5, label='Double Fit', zorder=4)
            
            g1_plot = amp1 * np.exp(-0.5 * ((x_plot - mu1) / sigma1)**2) + c
            g2_plot = amp2 * np.exp(-0.5 * ((x_plot - mu2) / sigma2)**2) + c
            ax.plot(x_plot, g1_plot, 'b--', alpha=0.8, linewidth=1.5, label='Source 1')
            ax.plot(x_plot, g2_plot, 'g--', alpha=0.8, linewidth=1.5, label='Source 2')
            
            ax.legend(fontsize='10', loc='best')
            ax.set_title(f"{label}: Peak Distance = {dist_mm:.1f} mm", color='blue', fontweight='bold', fontsize=12)
            return dist_mm, mu1, mu2
        except:
            is_single_peak = True # Se il fit doppio esplode, prova il singolo

    # ==========================================
    # FALLBACK: FIT SINGOLA GAUSSIANA
    # ==========================================
    if is_single_peak:
        idx_max = np.argmax(profile_counts)
        p0_single = [profile_counts[idx_max], axis_vals[idx_max], 3.0, 0]
        try:
            popt_s, _ = curve_fit(single_gaussian, axis_vals, profile_counts, p0=p0_single, maxfev=100000)
            x_plot = np.linspace(axis_vals[0], axis_vals[-1], 200)
            ax.plot(x_plot, single_gaussian(x_plot, *popt_s), 'r-', linewidth=2.5, label='Single Fit', zorder=4)
            ax.legend(fontsize='10', loc='best')
            ax.set_title(f"{label}: Sources Aligned (1 Peak)", color='black', fontweight='bold', fontsize=12)
            return 0.0, popt_s[1], None # 0.0 indica che non c'è distanza da misurare
        except:
            ax.set_title(f"{label}: Fit Failed", color='red', fontsize=12)
            return None, None, None

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
    
    print(f"\n📏 Calcolo Imaging Multi-Sorgente/Phantom su {run_name}...")

    x_mm, y_mm = load_scatter_data(run_path)
    if x_mm is None or len(x_mm) == 0:
        print("❌ Nessun dato spaziale trovato.")
        return

    bins_edge = np.arange(-13.0, 14.0, BIN_SIZE_MM)
    heatmap, xedges, yedges = np.histogram2d(x_mm, y_mm, bins=bins_edge)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    
    profile_x = np.sum(heatmap, axis=1) 
    profile_y = np.sum(heatmap, axis=0) 

    plt.rcParams.update({'font.size': 12}) 
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 2)

    ax_map = fig.add_subplot(gs[0, 0])
    im = ax_map.imshow(heatmap.T, origin='lower', extent=[-13, 13, -13, 13], cmap='plasma', interpolation='nearest')
    ax_map.set_title("2D Reconstructed Imaging", fontweight='bold')
    ax_map.set_xlabel("Position X [mm]")
    ax_map.set_ylabel("Position Y [mm]")
    plt.colorbar(im, ax=ax_map, label="Counts")

    # Profilo X
    ax_x = fig.add_subplot(gs[1, 0])
    dist_x, mux1, mux2 = fit_double_profile(x_centers, profile_x, "X-Axis Profile", ax_x)
    ax_x.set_xlabel("Position X [mm]")
    ax_x.set_ylabel("Counts")

    # Profilo Y
    ax_y = fig.add_subplot(gs[0, 1])
    dist_y, muy1, muy2 = fit_double_profile(y_centers, profile_y, "Y-Axis Profile", ax_y)
    ax_y.set_xlabel("Position Y [mm]")
    ax_y.set_ylabel("Counts")

    # Box Informativo Aggiornato per Fantocci
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis('off')
    
    info_text = f"PHANTOM / MULTI-SOURCE ANALYSIS\n\n"
    info_text += f"Run: {run_name}\n"
    info_text += f"Binning: {BIN_SIZE_MM} mm\n"
    info_text += f"----------------------------------\n\n"
    
    if dist_x is not None:
        if dist_x > 0:
            info_text += f"X-Axis: Resolved\n  Dist:  {dist_x:.2f} mm\n  Peaks: {mux1:.1f}, {mux2:.1f}\n\n"
        else:
            info_text += f"X-Axis: Aligned (Single Profile)\n  Center: {mux1:.1f} mm\n\n"
            
    if dist_y is not None:
        if dist_y > 0:
            info_text += f"Y-Axis: Resolved\n  Dist:  {dist_y:.2f} mm\n  Peaks: {muy1:.1f}, {muy2:.1f}\n"
        else:
            info_text += f"Y-Axis: Aligned (Single Profile)\n  Center: {muy1:.1f} mm\n"

    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.9, edgecolor='gray')
    ax_info.text(0.1, 0.5, info_text, fontsize=13, family='monospace', verticalalignment='center', bbox=props)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "Phantom_Imaging_Analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Analisi Phantom/Multi-Sorgente salvata in: {out_path}")

if __name__ == "__main__":
    main()