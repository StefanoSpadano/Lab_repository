import sys
import os
import json
import numpy as np
import matplotlib
# Backend non-interattivo
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
from analysis.io_manager import get_project_root

# --- CONFIGURAZIONE ---
TARGET_PAIRS = {
    "Total": "total_charge",
    "Cluster": "main_clust_charge"
}

# --- DEFINIZIONE MODELLO (CB + STEP, SENZA COSTANTE) ---

def crystal_ball(x, N, mu, sigma, alpha, n):
    if sigma <= 0: return np.zeros_like(x)
    t = (x - mu) / sigma
    if alpha < 0: alpha = -alpha
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

def step_function(x, S, mu_step, sigma_step):
    # Ora mu_step e sigma_step sono parametri liberi, non rigidamente legati al picco
    return S * 0.5 * erfc((x - mu_step) / (sigma_step * np.sqrt(2)))

def full_model(x, N, mu, sigma, alpha, n, S, mu_step, sigma_step, c):
    cb = crystal_ball(x, N, mu, sigma, alpha, n)
    step = step_function(x, S, mu_step, sigma_step)
    return cb + step + c

# ... (mantieni intatte le funzioni load_histogram_data e rebin_histogram) ...

def perform_step_fit(bins, counts, label, output_dir):
    print(f"  📐 Fit CB + Step + Const: {label}...")
    
    REBIN_FACTOR = 2
    bins_rebinned, counts_rebinned = rebin_histogram(bins, counts, REBIN_FACTOR)
    centers = 0.5 * (bins_rebinned[:-1] + bins_rebinned[1:])
    
    # --- ROI SEARCH ---
    SEARCH_MIN = 3000 
    SEARCH_MAX = 8000 
    
    mask_search = (centers > SEARCH_MIN) & (centers < SEARCH_MAX)
    if np.sum(mask_search) == 0: return None

    idx_max = np.argmax(counts_rebinned[mask_search])
    mu_guess = centers[mask_search][idx_max]
    amp_guess = counts_rebinned[mask_search][idx_max]
    
    # --- FINESTRE DI FIT STRINGENTI ---
    # Allarghiamo leggermente a 1500 per compensare il REBIN a 8
    FIT_WINDOW_LEFT = 1200  
    FIT_WINDOW_RIGHT = 800 
    
    roi_min = mu_guess - FIT_WINDOW_LEFT
    roi_max = mu_guess + FIT_WINDOW_RIGHT
    
    mask_fit = (centers >= roi_min) & (centers <= roi_max)
    x_fit = centers[mask_fit]
    y_fit = counts_rebinned[mask_fit]
    
    # Abbassiamo il limite di sicurezza da 15 a 10 (il minimo per 9 parametri)
    if len(x_fit) < 10: 
        print(f"    ⚠️ ABORT: Troppi pochi punti nella ROI ({len(x_fit)}).")
        return None

    # --- GUESS INTELLIGENTE ---
    left_level = np.mean(y_fit[:10]) 
    step_guess = left_level if left_level > 0 else amp_guess * 0.1

    # Stima Sigma classica per la Crystal Ball
    half_max = amp_guess / 2.0
    idxs_above = np.where(y_fit > half_max)[0]
    if len(idxs_above) > 1:
        width_adu = x_fit[idxs_above[-1]] - x_fit[idxs_above[0]]
        sigma_guess = width_adu / 2.355
    else:
        sigma_guess = 80.0 
    
    # Protezioni Sigma
    if sigma_guess < 10: sigma_guess = 50.0
    if sigma_guess > 400: sigma_guess = 350.0 # Sicuro dentro il bound di 400

    # Guess Scalino
    mu_step_guess = mu_guess - (2 * sigma_guess)
    sigma_step_guess = sigma_guess

    # Guess Costante: Deve essere strettamente dentro i bounds
    c_max_bound = amp_guess * 0.05 
    c_guess_raw = np.mean(y_fit[-10:])
    
    # Se il fondo reale è più alto del 5%, costringiamo il guess al 4.9% per non sforare
    if c_guess_raw > c_max_bound:
        c_guess = c_max_bound * 0.98
    elif c_guess_raw < 0:
        c_guess = 0.0
    else:
        c_guess = c_guess_raw

    # Assicuriamoci che mu_step_guess sia dentro i bounds
    # Bounds_high per mu_step è mu_guess - 50.
    if mu_step_guess >= (mu_guess - 50):
        mu_step_guess = mu_guess - 60

    # p0 = [N, mu, sigma, alpha, n, S, mu_step, sigma_step, c]
    p0 = [amp_guess, mu_guess, sigma_guess, 1.5, 2.0, step_guess, mu_step_guess, sigma_step_guess, c_guess]
    
    # --- BOUNDS (Limiti Fisici Rigidi) ---
    bounds_low  = [0,      mu_guess-200, 10,  0.1, 1.01,  0,      roi_min,      10,  0]
    bounds_high = [np.inf, mu_guess+200, 400, 5.0, 100.0, np.inf, mu_guess-50, 500, c_max_bound]
    
    # --- PROTEZIONE ESTREMA (Debug) ---
    # Controlliamo al volo se p0 è fuori dai bounds e in caso stampiamo il colpevole
    for i in range(len(p0)):
        if p0[i] < bounds_low[i] or p0[i] > bounds_high[i]:
            print(f"  ⚠️ Warning: Parametro {i} (val={p0[i]:.2f}) fuori dai bounds [{bounds_low[i]:.2f}, {bounds_high[i]:.2f}]. Forzo al centro.")
            # Se è fuori, lo forziamo a metà tra il limite basso e alto per farlo partire
            p0[i] = bounds_low[i] + (bounds_high[i] - bounds_low[i]) / 2.0

    try:
        popt, pcov = curve_fit(full_model, x_fit, y_fit, p0=p0, bounds=(bounds_low, bounds_high), maxfev=500000)
    except Exception as e:
        print(f"  ❌ Fit fallito: {e}")
        return None

    N_fit, mu_fit, sigma_fit, alpha_fit, n_fit, S_fit, mu_step_fit, sigma_step_fit, c_fit = popt
    
    fwhm = 2.355 * sigma_fit
    res_percent = (fwhm / mu_fit) * 100
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 8)) 
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    ax0 = fig.add_subplot(gs[0])
    
    ax0.step(centers, counts_rebinned, where='mid', color='lightgray', linewidth=1, label='Full Spectrum')
    ax0.step(x_fit, y_fit, where='mid', color='black', alpha=0.8, linewidth=1.5, label='Fit Region')
    
    x_plot = np.linspace(x_fit[0], x_fit[-1], 1000)
    y_total = full_model(x_plot, *popt)
    ax0.plot(x_plot, y_total, 'r-', linewidth=2, label=f'Fit Model')

    y_cb = crystal_ball(x_plot, N_fit, mu_fit, sigma_fit, alpha_fit, n_fit)
    y_bkg_vis = step_function(x_plot, S_fit, mu_step_fit, sigma_step_fit) + c_fit
    
    ax0.plot(x_plot, y_cb, 'g--', alpha=0.6, label='Crystal Ball')
    ax0.plot(x_plot, y_bkg_vis, 'orange', linestyle='--', alpha=0.8, label='Bkg (Step + Const)')

    x_max_view = mu_fit + 3000 
    if x_max_view > centers[-1]: x_max_view = centers[-1]
    ax0.set_xlim(0, x_max_view) 
    
    y_max_roi = np.max(y_fit) * 1.2
    ax0.set_ylim(0, y_max_roi)

    ax0.set_ylabel("Counts")
    ax0.set_title(f"Fit Analysis: {label}\nResolution: {res_percent:.2f}% @ {mu_fit:.0f} ADC")
    ax0.legend()
    ax0.grid(alpha=0.3)
    
    # --- RESIDUI COLORATI ---
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    
    model_at_points = full_model(x_fit, *popt)
    residuals = y_fit - model_at_points
    
    model_safe = model_at_points.copy()
    model_safe[model_safe < 1] = 1
    dof = len(x_fit) - 9 # Ora abbiamo 9 parametri
    chi2_global = np.sum(residuals**2/model_safe) / dof if dof > 0 else 0
    
    peak_mask = (x_fit >= mu_fit - 2.0*sigma_fit) & (x_fit <= mu_fit + 2.0*sigma_fit)
    
    ax1.scatter(x_fit[~peak_mask], residuals[~peak_mask], color='purple', s=10, alpha=0.6, label='Bkg Tails')
    ax1.scatter(x_fit[peak_mask], residuals[peak_mask], color='red', s=20, label='Peak Area')
    
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_xlabel("ADC Channel")
    ax1.set_ylabel("Residuals")
    
    stats_text = r"Global $\chi^2/ndf$: " + f"{chi2_global:.2f}"
    ax1.text(0.02, 0.1, stats_text, transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"step_fit_{label}_REBIN_{REBIN_FACTOR}.png")
    plt.savefig(out_file)
    plt.close()
    
    print(f"    📸 IMMAGINE SALVATA IN: {out_file}")
    print(f"       -> Chi2 calcolato: {chi2_global:.2f}")
    
    return {
        "res": res_percent,
        "chi2": chi2_global,
        "mu": mu_fit,
        "sigma": sigma_fit,
        "amp": N_fit,
        "step_amp": S_fit,
        "const": c_fit
    }

def load_histogram_data(run_path, hist_name):
    json_path = os.path.join(run_path, "histograms", f"{hist_name}.json")
    if not os.path.exists(json_path): 
        print(f"    ⚠️ File non trovato: {json_path}")
        return None, None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "bins" in data:
                return np.array(data["bins"]), np.array(data["counts"])
            elif isinstance(data, list) and len(data) == 2:
                return np.array(data[0]), np.array(data[1])
    except Exception as e: 
        print(f"    ❌ Errore lettura JSON {hist_name}: {e}")
        return None, None
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

def main():
    print("\n🚨🚨🚨 STO ESEGUENDO IL FILE MODIFICATO! 🚨🚨🚨\n") 
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.fit_scipy_step <date_folder>")
        return
    
    date_str = sys.argv[1]
    root_dir = get_project_root()
    base_res_dir = os.path.join(root_dir, "Results", date_str)
    
    summary_file = os.path.join(base_res_dir, "thesis_summary_data_step.json")
    
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                summary_data = json.load(f)
            print(f"📂 Caricato database Step esistente ({len(summary_data)} runs).")
        except: summary_data = {}
    else: summary_data = {}

    runs = [d for d in os.listdir(base_res_dir) if d.startswith("Run") and os.path.isdir(os.path.join(base_res_dir, d))]
    runs.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

    for run in runs:
        print(f"\n--- Analisi CB+Step {run} ---")
        run_path = os.path.join(base_res_dir, run)
        fit_out_dir = os.path.join(run_path, "fits_step")
        
        run_results = summary_data.get(run, {})
        updated_any = False
        
        for label, json_name in TARGET_PAIRS.items():
            bins, counts = load_histogram_data(run_path, json_name)
            if bins is not None:
                res = perform_step_fit(bins, counts, label, fit_out_dir)
                if res: 
                    run_results[label] = res
                    updated_any = True
            else:
                print(f"    Passaggio saltato per {label} (dati mancanti).")
        
        if updated_any:
             summary_data[run] = run_results

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"\n✅ Fit Step completati. Dati in: {summary_file}")

if __name__ == "__main__":
    main()