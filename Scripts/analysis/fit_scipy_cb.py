import sys
import os
import json
import numpy as np
import matplotlib
# Backend non-interattivo: Fondamentale per evitare crash su Windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from analysis.io_manager import get_project_root

# --- CONFIGURAZIONE ---
TARGET_PAIRS = {
    "Total": "total_charge",
    "Cluster": "main_clust_charge"
}

# --- DEFINIZIONE CRYSTAL BALL ---
def crystal_ball(x, N, mu, sigma, alpha, n):
    if sigma <= 0: return np.zeros_like(x)
    t = (x - mu) / sigma
    if alpha < 0: alpha = -alpha
    # Vincolo soft interno
    if n <= 1: n = 1.01 
    if alpha == 0: alpha = 0.01

    A = (n / abs(alpha))**n * np.exp(-0.5 * abs(alpha)**2)
    B = (n / abs(alpha)) - abs(alpha)
    
    conditions = [t > -alpha, t <= -alpha]
    functions = [
        lambda t: np.exp(-0.5 * t**2),
        lambda t: A * (B - t)**(-n)
    ]
    return N * np.piecewise(t, conditions, functions)

def cb_plus_linear(x, N, mu, sigma, alpha, n, m, c):
    # m = slope, c = intercept
    return crystal_ball(x, N, mu, sigma, alpha, n) + (m * x + c)

def load_histogram_data(run_path, hist_name):
    json_path = os.path.join(run_path, "histograms", f"{hist_name}.json")
    if not os.path.exists(json_path): return None, None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "bins" in data:
                return np.array(data["bins"]), np.array(data["counts"])
            elif isinstance(data, list) and len(data) == 2:
                return np.array(data[0]), np.array(data[1])
    except: return None, None
    return None, None

def rebin_histogram(bins, counts, rebin_factor):
    if rebin_factor <= 1: return bins, counts
    n_bins_new = len(counts) // rebin_factor
    limit = n_bins_new * rebin_factor
    counts_trunc = counts[:limit]
    bins_trunc = bins[:limit+1]
    new_counts = counts_trunc.reshape(-1, rebin_factor).sum(axis=1)
    new_bins = bins_trunc[::rebin_factor]
    return new_bins, new_counts

def perform_cb_fit(bins, counts, label, output_dir):
    print(f"   🔮 Crystal Ball Fit (Physics Constrained): {label}...")
    
    REBIN_FACTOR = 4
    bins, counts = rebin_histogram(bins, counts, REBIN_FACTOR)
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    # --- ROI SEARCH ---
    # MIN alto per saltare il fantoccio (scattering)
    # MAX alto per non perdere il picco Total che shifta a destra
    SEARCH_MIN = 3000 
    SEARCH_MAX = 8000 
    
    mask_search = (centers > SEARCH_MIN) & (centers < SEARCH_MAX)
    if np.sum(mask_search) == 0: 
        print("     -> Nessun dato nella finestra di ricerca.")
        return None

    idx_max = np.argmax(counts[mask_search])
    mu_guess = centers[mask_search][idx_max]
    amp_guess = counts[mask_search][idx_max]
    
    # --- FINESTRE DI FIT (SIMMETRICHE PER STABILITÀ) ---
    # 1200/1200 è la configurazione 'salvagente' che funziona sempre
    FIT_WINDOW_LEFT = 800
    FIT_WINDOW_RIGHT = 1500
    
    roi_min = mu_guess - FIT_WINDOW_LEFT
    roi_max = mu_guess + FIT_WINDOW_RIGHT
    
    mask_fit = (centers >= roi_min) & (centers <= roi_max)
    x_fit = centers[mask_fit]
    y_fit = counts[mask_fit]
    
    if len(x_fit) < 15: 
        print(f"     -> Troppi pochi punti per il fit ({len(x_fit)})")
        return None

    # --- GUESS INTELLIGENTE ---
    slope_guess = 0.0
    intercept_guess = np.min(y_fit)

    half_max = amp_guess / 2.0
    idxs_above = np.where(y_fit > half_max)[0]
    if len(idxs_above) > 1:
        width_adu = x_fit[idxs_above[-1]] - x_fit[idxs_above[0]]
        sigma_guess = width_adu / 2.355
    else:
        sigma_guess = 100.0 

    if sigma_guess < 10: sigma_guess = 50.0
    if sigma_guess > 600: sigma_guess = 400.0

    # P0: [N, mu, sigma, alpha, n, slope, intercept]
    p0 = [amp_guess, mu_guess, sigma_guess, 1.5, 2.0, slope_guess, intercept_guess]
    
    # BOUNDS: slope <= 0 (fondo piatto o discesa), n > 1.01 (coda fisica)
    bounds_low  = [0,      roi_min, 10,   0.1, 1.01,  -np.inf, -np.inf]
    bounds_high = [np.inf, roi_max, 600,  5.0, 100.0, 0.00001, np.inf]
    
    try:
        popt, pcov = curve_fit(cb_plus_linear, x_fit, y_fit, p0=p0, bounds=(bounds_low, bounds_high), maxfev=2000000)
    except Exception as e:
        print(f"   ❌ Fit fallito: {e}")
        return None

    N_fit, mu_fit, sigma_fit, alpha_fit, n_fit, m_fit, c_fit = popt
    
    fwhm = 2.355 * sigma_fit
    res_percent = (fwhm / mu_fit) * 100
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = fig.add_subplot(gs[0])
    
    ax0.step(x_fit, y_fit, where='mid', color='black', alpha=0.7, label='Dati')
    x_plot = np.linspace(x_fit[0], x_fit[-1], 1000)
    y_total = cb_plus_linear(x_plot, *popt)
    y_bkg = m_fit * x_plot + c_fit
    
    ax0.plot(x_plot, y_total, 'r-', label=f'Crystal Ball (R={res_percent:.2f}%)')
    ax0.plot(x_plot, y_bkg, 'b--', label=f'Fondo (m={m_fit:.4f})')
    
    ax0.set_title(f"Fit Crystal Ball: {label}\n$\\alpha$={alpha_fit:.2f}, $n$={n_fit:.2f}")
    ax0.legend()
    ax0.grid(alpha=0.3)
    
    model_at_points = cb_plus_linear(x_fit, *popt)
    residuals = y_fit - model_at_points
    
    model_safe = model_at_points.copy()
    model_safe[model_safe < 1] = 1
    dof = len(x_fit) - 7
    chi2 = np.sum(residuals**2/model_safe) / dof if dof > 0 else 0
    
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.scatter(x_fit, residuals, color='green', s=15)
    ax1.axhline(0, color='black', linestyle='--')
    
    # Fix sintassi LaTeX (raw string r"")
    stats_text = r"$\chi^2/ndf$: " + f"{chi2:.2f}"
    ax1.text(0.02, 0.1, stats_text, transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"cb_fit_{label}.png")
    plt.savefig(out_file)
    plt.close()
    
    print(f"      🎯 RISULTATO MU: {mu_fit:.4f} (Sigma: {sigma_fit:.2f})")

    return {
        "res": res_percent,
        "chi2": chi2,
        "mu": mu_fit,
        "sigma": sigma_fit,
        "amp": N_fit
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.fit_scipy_cb <date_folder>")
        return
    
    date_str = sys.argv[1]
    root_dir = get_project_root()
    base_res_dir = os.path.join(root_dir, "Results", date_str)
    
    # FILE SUMMARY CONDIVISO
    summary_file = os.path.join(base_res_dir, "thesis_summary_data.json")
    
    # 1. CARICA DATI ESISTENTI (INCREMENTAL UPDATE)
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                summary_data = json.load(f)
            print(f"📂 Caricato database esistente ({len(summary_data)} runs).")
        except json.JSONDecodeError:
            summary_data = {}
    else:
        summary_data = {}

    runs = [d for d in os.listdir(base_res_dir) if d.startswith("Run") and os.path.isdir(os.path.join(base_res_dir, d))]
    runs.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

    for run in runs:
        print(f"\n--- Analisi Crystal Ball {run} ---")
        run_path = os.path.join(base_res_dir, run)
        fit_out_dir = os.path.join(run_path, "fits_cb")
        
        # Recupera dati esistenti per questa run o crea nuovo dict
        run_results = summary_data.get(run, {})
        
        updated_any = False
        for label, json_name in TARGET_PAIRS.items():
            bins, counts = load_histogram_data(run_path, json_name)
            if bins is not None:
                res = perform_cb_fit(bins, counts, label, fit_out_dir)
                if res: 
                    run_results[label] = res
                    updated_any = True
        
        if updated_any:
             summary_data[run] = run_results
             print(f"   -> Dati aggiornati per {run}")
        else:
             print(f"   ⚠️ Nessun fit valido per {run}")

    # 2. SALVA TUTTO
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"\n✅ Dati riassuntivi salvati in: {summary_file}")

if __name__ == "__main__":
    main()