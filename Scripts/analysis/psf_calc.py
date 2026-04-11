import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend sicuro
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analysis.io_manager import get_project_root

# --- CONFIGURAZIONE PSF ---
BIN_SIZE_MM = 3.2     # Risoluzione della nostra ricostruzione
ROI_WINDOW_MM = 30.0    # Quanto ci allarghiamo dal picco per il fit
DETECTOR_SIZE_MM = 26.0 # Dimensione fisica approssimativa
HISTO_BIN_MM = 0.5

def gaussian(x, amp, mu, sigma, c):
    """Gaussiana spaziale + fondo costante"""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + c

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
    return None, None

def fit_profile(axis_vals, profile_counts, label, ax):
    """
    Fit con SMART GUESS (Momenti Pesati).
    Ora estrae e restituisce anche l'incertezza sulla FWHM.
    """
    idx_max = np.argmax(profile_counts)
    mu_rough = axis_vals[idx_max]
    
    mask = (axis_vals > mu_rough - ROI_WINDOW_MM) & (axis_vals < mu_rough + ROI_WINDOW_MM)
    x_fit = axis_vals[mask]
    y_fit = profile_counts[mask]
    
    ax.step(axis_vals, profile_counts, where='mid', color='gray', alpha=0.5, label='Data', linewidth=1.5)
    ax.grid(alpha=0.3)
    
    if len(x_fit) < 4: 
        ax.text(0.5, 0.5, "Not enough points", transform=ax.transAxes, ha='center')
        return None, None

    if np.max(profile_counts) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
        return None, None

    ax.scatter(x_fit, y_fit, color='black', s=20, label='Fit Points', zorder=3)

    try:
        mean_val = np.average(x_fit, weights=y_fit)
        variance = np.average((x_fit - mean_val)**2, weights=y_fit)
        sigma_val = np.sqrt(variance)
        amp_val = np.max(y_fit) 
    except:
        mean_val = mu_rough
        sigma_val = 2.0
        amp_val = np.max(y_fit)

    p0 = [amp_val, mean_val, sigma_val, 0] 
    bounds_low = [0, -np.inf, 0.01, 0]
    bounds_high = [np.inf, np.inf, 50.0, np.inf]
    
    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, bounds=(bounds_low, bounds_high), maxfev=1000000)
        # Estrazione Errori dalla matrice di covarianza
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Fit fallito per {label}: {e}")
        ax.text(0.5, 0.5, "Fit Failed", transform=ax.transAxes, color='red', ha='center')
        return None, None
        
    amp, mu, sigma, c = popt
    sigma_err = perr[2] # Incertezza su sigma (3° parametro)
    
    fwhm = 2.355 * abs(sigma)
    fwhm_err = 2.355 * sigma_err
    
    x_plot = np.linspace(x_fit[0], x_fit[-1], 200)
    ax.plot(x_plot, gaussian(x_plot, *popt), 'r-', linewidth=2.5, label='Gaussian Fit', zorder=4)
    ax.legend(fontsize='10')
    
    if fwhm < 0.5: 
        title_color = 'red'
        status = " (Pixel Locked)"
    else:
        title_color = 'black'
        status = ""
        
    ax.set_title(f"{label}: FWHM = {fwhm:.2f} ± {fwhm_err:.2f} mm{status}", color=title_color, fontweight='bold', fontsize=12)
    
    return fwhm, fwhm_err

def process_run(date_str, run_name, root_dir):
    """Elabora la singola run"""
    print(f"\n🔭 Analisi PSF su {run_name}...")
    run_path = os.path.join(root_dir, "Results", date_str, run_name)
    out_dir = os.path.join(run_path, "psf_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    x_mm, y_mm = load_scatter_data(run_path)
    if x_mm is None or len(x_mm) == 0:
        print("❌ Nessun dato spaziale trovato.")
        return

    bins_edge = np.arange(-13.0, 13.0, BIN_SIZE_MM)
    heatmap, xedges, yedges = np.histogram2d(x_mm, y_mm, bins=bins_edge)
    
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    idx_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_x_idx, peak_y_idx = idx_max
    peak_x_mm = x_centers[peak_x_idx]
    peak_y_mm = y_centers[peak_y_idx]

    profile_x = heatmap[:, peak_y_idx]
    profile_y = heatmap[peak_x_idx, :]

    # Ingrandito il font di default per la presentazione
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 2)

    # --- A. HEATMAP 2D ---
    ax_map = fig.add_subplot(gs[0, 0])
    im = ax_map.imshow(heatmap.T, origin='lower', extent=[-13, 13, -13, 13], cmap='plasma', interpolation='nearest')
    ax_map.set_title("2D Reconstructed Image", fontweight='bold')
    ax_map.set_xlabel("Position X [mm]")
    ax_map.set_ylabel("Position Y [mm]")
    plt.colorbar(im, ax=ax_map, label="Counts")
    ax_map.axvline(peak_x_mm, color='white', linestyle='--', alpha=0.6, linewidth=1.5)
    ax_map.axhline(peak_y_mm, color='white', linestyle='--', alpha=0.6, linewidth=1.5)

    # --- B. PROFILO X & Y ---
    ax_x = fig.add_subplot(gs[1, 0])
    fwhm_x, err_x = fit_profile(x_centers, profile_x, "X-Axis Profile", ax_x)
    ax_x.set_xlabel("Position X [mm]")
    ax_x.set_ylabel("Counts") # AGGIUNTO

    ax_y = fig.add_subplot(gs[0, 1])
    fwhm_y, err_y = fit_profile(y_centers, profile_y, "Y-Axis Profile", ax_y)
    ax_y.set_xlabel("Position Y [mm]")
    ax_y.set_ylabel("Counts") # AGGIUNTO

    # --- INFO BOX ---
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis('off')
    
    info_text = f"POINT SPREAD FUNCTION (PSF)\n\n"
    
    if fwhm_x and fwhm_y:
        info_text += f"X-Axis FWHM :  {fwhm_x:.2f} ± {err_x:.2f} mm\n"
        info_text += f"Y-Axis FWHM :  {fwhm_y:.2f} ± {err_y:.2f} mm\n"
        
        psf_mean = (fwhm_x + fwhm_y) / 2.0
        # Propagazione errore media: 0.5 * sqrt(err_x^2 + err_y^2)
        psf_mean_err = 0.5 * np.sqrt(err_x**2 + err_y**2)
        
        info_text += f"---------------------------------\n"
        info_text += f"Mean PSF    :  {psf_mean:.2f} ± {psf_mean_err:.2f} mm\n"
        
        print(f"   ---> Asse X: {fwhm_x:.2f} ± {err_x:.2f} mm")
        print(f"   ---> Asse Y: {fwhm_y:.2f} ± {err_y:.2f} mm")
        print(f"   🎯 PSF MEDIA: {psf_mean:.2f} ± {psf_mean_err:.2f} mm")
    else:
        info_text += "Fit Failed on at least one axis."
        print("   ⚠️ Fit fallito su almeno un asse.")

    # Box testo stilizzato per essere leggibile proiettato
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.9, edgecolor='gray')
    ax_info.text(0.1, 0.5, info_text, fontsize=14, family='monospace', verticalalignment='center', bbox=props)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "PSF_Analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.calc_psf <date> [run_name]")