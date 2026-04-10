import sys
import os
import json
import numpy as np
import matplotlib
# Non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
from analysis.io_manager import get_project_root

# --- CONFIGURATION ---
TARGET_PAIRS = {
    "Total": "total_charge",
    "Cluster": "main_clust_charge"
}
# ----------------------

def load_histogram_data(run_path, hist_name):
    json_path = os.path.join(run_path, "histograms", f"{hist_name}.json")
    if not os.path.exists(json_path):
        return None, None
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
    if rebin_factor <= 1:
        return bins, counts
    n_bins_new = len(counts) // rebin_factor
    if n_bins_new == 0:                        
        return np.array([]), np.array([])
    limit = n_bins_new * rebin_factor
    counts_trunc = counts[:limit]
    bins_trunc = bins[:limit+1]
    new_counts = counts_trunc.reshape(-1, rebin_factor).sum(axis=1)
    new_bins = bins_trunc[::rebin_factor]
    return new_bins, new_counts

def calculate_resolution(mu, sigma):
    if mu <= 0: return 0.0
    fwhm = 2.355 * sigma
    return (fwhm / mu) * 100

def perform_advanced_fit(bins, counts, label, output_dir):
    print(f"   GMM Fitting (4 Comp): {label}...")
    
    REBIN_FACTOR = 4  
    bins_reb, counts_reb = rebin_histogram(bins, counts, REBIN_FACTOR)
    centers = 0.5 * (bins_reb[:-1] + bins_reb[1:])
    
    threshold = 50 
    mask = (centers > threshold) & (counts_reb > 0)
    x_clean = centers[mask]
    y_clean = counts_reb[mask]
    
    if len(x_clean) < 20: return None

    # Weighted dataset for GMM
    scale_factor = 20000 / np.sum(y_clean) if np.sum(y_clean) > 20000 else 1
    y_scaled = (y_clean * scale_factor).astype(int)
    samples = []
    for val, count in zip(x_clean, y_scaled):
        samples.extend([val] * count)
    samples = np.array(samples).reshape(-1, 1)

    # Effective total events calculated on clean raw counts
    n_tot_events = np.sum(y_clean)

    # 4 Components for Barium (Multipeak + background)
    N_COMPONENTS = 4
    try:
        gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42, max_iter=200, n_init=5)
        gmm.fit(samples)
    except: return None
    
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    idx = np.argsort(means)
    means = means[idx]; covariances = covariances[idx]; weights = weights[idx]

    sigmas = np.sqrt(covariances)
    peak_heights = weights / sigmas
    
    # --- PEAK SEARCH PARAMETERS ---
    MIN_CH_PICCO = 3000   
    MAX_CH_PICCO = 7000  
    
    valid_indices = [
        i for i, m in enumerate(means) 
        if MIN_CH_PICCO < m < MAX_CH_PICCO
    ]
    
    if valid_indices:
        best_valid_idx = valid_indices[np.argmax(peak_heights[valid_indices])]
        photopeak_idx = best_valid_idx
    else:
        photopeak_idx = np.argmax(peak_heights)
        
    mu_peak = means[photopeak_idx]
    sigma_peak = sigmas[photopeak_idx]
    weight_peak = weights[photopeak_idx]
    
    photopeak_res = calculate_resolution(mu_peak, sigma_peak)

    # =========================================================================
    # --- STATISTICAL ERROR CALCULATION ---
    # =========================================================================
    n_eff_peak = weight_peak * n_tot_events
    
    if n_eff_peak > 0:
        delta_mu = sigma_peak / np.sqrt(n_eff_peak)
        delta_sigma = sigma_peak / np.sqrt(2 * n_eff_peak)
        
        fwhm_peak = 2.355 * sigma_peak
        delta_fwhm = 2.355 * delta_sigma
        
        # Error propagation on resolution R = FWHM / mu
        if mu_peak > 0:
            delta_res = photopeak_res * np.sqrt((delta_fwhm / fwhm_peak)**2 + (delta_mu / mu_peak)**2)
        else:
            delta_res = 0.0
    else:
        delta_mu = delta_sigma = delta_fwhm = delta_res = 0.0

    print(f"      ---> {label} | FWHM: {2.355*sigma_peak:.2f} ± {delta_fwhm:.2f} ADU")
    print(f"      ---> {label} | Res : {photopeak_res:.2f} ± {delta_res:.2f} %")
    # =========================================================================

    # --- PLOTTING ---
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    x_plot = np.linspace(bins_reb[0], bins_reb[-1], 1000)
    norm_factor = np.sum(counts_reb * (bins_reb[1] - bins_reb[0]))
    
    # Calculate residuals
    model_y = np.zeros_like(x_clean)
    for i in range(len(means)):
        mu, sigma, w = means[i], np.sqrt(covariances[i]), weights[i]
        pdf = (w / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_clean - mu) / sigma)**2)
        model_y += pdf * norm_factor
    residuals = y_clean - model_y

    # Chi-Square ROI
    roi_min = mu_peak - 2.5 * sigma_peak
    roi_max = mu_peak + 2.5 * sigma_peak
    mask_roi = (x_clean >= roi_min) & (x_clean <= roi_max)
    
    chi_reduced = 0
    if np.sum(mask_roi) > 5:
        y_roi = y_clean[mask_roi]
        model_roi = model_y[mask_roi]
        model_roi_safe = model_roi.copy()
        model_roi_safe[model_roi_safe < 1] = 1 
        chi_sq_local = np.sum((y_roi - model_roi)**2 / model_roi_safe)
        ndf_local = len(y_roi) - 3 
        chi_reduced = chi_sq_local / ndf_local if ndf_local > 0 else 0

    # UPPER Panel
    ax0 = plt.subplot(gs[0])
    ax0.step(centers, counts_reb, where='mid', color='black', alpha=0.4, label=f'Data (Rebin x{REBIN_FACTOR})')
    
    total_pdf_plot = np.zeros_like(x_plot)
    colors = ['green', 'blue', 'orange', 'purple', 'cyan']
    
    for i in range(len(means)):
        mu, sigma, w = means[i], np.sqrt(covariances[i]), weights[i]
        pdf = (w / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_plot - mu) / sigma)**2)
        pdf_scaled = pdf * norm_factor
        total_pdf_plot += pdf_scaled
        
        col = colors[i % len(colors)]
        lbl_prefix = "★ PEAK" if i == photopeak_idx else f"Comp {i+1}"
        ax0.plot(x_plot, pdf_scaled, '--', color=col, linewidth=1.5, label=f"{lbl_prefix} ({mu:.0f} ADU)")

    ax0.plot(x_plot, total_pdf_plot, 'r-', linewidth=2, label='Total Fit (GMM)')
    ax0.set_title(f"GMM Fit (4 Comp): {label}", fontweight='bold', fontsize=14)
    ax0.set_ylabel("Counts", fontsize=12) # ADDED Y-AXIS LABEL
    ax0.legend(loc='upper right')
    ax0.grid(alpha=0.3)
    if "Total" in label: ax0.set_xlim(0, 8000)

    # ADDED TEXT BOX WITH ERRORS
    fit_info_text = (
        f"Photopeak Results:\n"
        f"$\\mu = {mu_peak:.1f} \\pm {delta_mu:.1f}$ ADU\n"
        f"FWHM $= {fwhm_peak:.1f} \\pm {delta_fwhm:.1f}$ ADU\n"
        f"Res. $= {photopeak_res:.2f} \\pm {delta_res:.2f}\\%$ "
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax0.text(0.02, 0.95, fit_info_text, transform=ax0.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    # LOWER Panel
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.scatter(x_clean, residuals, s=15, color='purple', alpha=0.5)
    ax1.scatter(x_clean[mask_roi], residuals[mask_roi], s=25, color='red', label='Peak ROI')
    ax1.axhline(0, color='black', linestyle='--')
    
    stats_text = r"Local $\chi^2$/ndf = " + f"{chi_reduced:.2f}"
    props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.08, stats_text, transform=ax1.transAxes, bbox=props2)
    
    ax1.set_xlabel("Energy [ADC Channels]", fontsize=12) # ADDED X-AXIS LABEL WITH UNITS
    ax1.set_ylabel("Residuals", fontsize=12)
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    final_out_dir = os.path.join(output_dir, "fits_gmm") 
    os.makedirs(final_out_dir, exist_ok=True)
    out_file = os.path.join(final_out_dir, f"gmm_fit_{label}.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    
    return {
        "mu": mu_peak,
        "sigma": sigma_peak,
        "res": photopeak_res,
        "chi2": chi_reduced
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.fit_scipy_gmm <date_folder>")
        return
    date_str = sys.argv[1]
    root_dir = get_project_root()
    base_res_dir = os.path.join(root_dir, "Results", date_str)
    
    runs = [d for d in os.listdir(base_res_dir) if d.startswith("Run") and os.path.isdir(os.path.join(base_res_dir, d))]
    runs.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

    # JSON DEDICATED TO GMM
    json_path = os.path.join(base_res_dir, "thesis_summary_data_gmm.json")
    
    # 1. LOAD EXISTING DATA (INCREMENTAL UPDATE)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                summary_data_gmm = json.load(f)
            print(f" Loaded existing GMM database ({len(summary_data_gmm)} runs).")
        except json.JSONDecodeError:
            summary_data_gmm = {}
    else:
        summary_data_gmm = {}

    for run in runs:
        print(f"\n--- GMM Analysis {run} ---")
        run_path = os.path.join(base_res_dir, run)
        
        # Retrieve data or create new
        run_results = summary_data_gmm.get(run, {})
        
        updated_any = False
        for label, json_name in TARGET_PAIRS.items():
            bins, counts = load_histogram_data(run_path, json_name)
            if bins is not None:
                res = perform_advanced_fit(bins, counts, label, run_path) 
                if res: 
                    run_results[label] = res
                    updated_any = True

        if updated_any:
            summary_data_gmm[run] = run_results
            print(f"   -> GMM Data updated for {run}")

    # 2. SAVING
    with open(json_path, "w") as f:
        json.dump(summary_data_gmm, f, indent=4)
        
    print(f"\n GMM summary data saved in: {json_path}")

if __name__ == "__main__":
    main()