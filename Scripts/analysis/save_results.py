import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.io_manager import get_run_output_dir

# Clean Matplotlib style
plt.style.use('seaborn-v0_8-paper')

def convert_numpy(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def save_all_results(results, date_str, run_name):
    """
    Saves the results replicating the Legacy plots (npix, xyglob 2D, etc.)
    """
    base_dir = get_run_output_dir(date_str, run_name)
    dirs = {
        "hist": os.path.join(base_dir, "histograms"),
        "maps": os.path.join(base_dir, "maps"),
        "plots": os.path.join(base_dir, "plots")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    print(f" Saving Legacy style Plots...")

    # 1D HISTOGRAMS (npix, charge, xglob 1D)
    if "histograms" in results:
        for name, (bins, counts) in results["histograms"].items():
            # JSON Data
            with open(os.path.join(dirs["hist"], f"{name}.json"), "w") as f:
                json.dump({"bins": convert_numpy(bins), "counts": convert_numpy(counts)}, f)
            
            # PNG Plot
            plt.figure(figsize=(8, 6))
            centers = 0.5 * (bins[:-1] + bins[1:])
            plt.step(centers, counts, where='mid', color='royalblue', linewidth=1.5, label=name)
            plt.fill_between(centers, counts, step='mid', alpha=0.3, color='royalblue')
            
            # Zoom for charge spectra
            if "charge" in name:
                nonzero = np.where(counts > 0)[0]
                if len(nonzero) > 0: plt.xlim(0, bins[nonzero[-1]] * 1.1)
            # Fixed limits for positions
            if name in ['xglob', 'yglob', 'xclus', 'yclus']:
                plt.xlim(-12.8, 12.8)
            
            plt.title(f"{name}_sub")
            plt.xlabel("Value")
            plt.ylabel("Counts")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(dirs["plots"], f"{name}_sub.png"))
            plt.close()

    # 2D MAPS & SCATTER DATA
    if "scatter_data" in results:
        sc = results["scatter_data"]

        # Without this, calc_psf.py has no data to work with!
        scatter_json_path = os.path.join(dirs["hist"], "scatter_data.json")
        try:
            # Convert the entire numpy dictionary to serializable lists
            serializable_sc = {k: convert_numpy(v) for k, v in sc.items()}
            with open(scatter_json_path, "w") as f:
                json.dump(serializable_sc, f)
            print(f"   -> Saved scatter_data.json (for PSF)")
        except Exception as e:
            print(f"   Error saving scatter JSON: {e}")

        # 2D Plot Definitions
        plot_defs = [
            ("xyglob_sub.png", "xyglob_x", "xyglob_y", 12, [[-12.8, 12.8], [-12.8, 12.8]]),
            ("xyclus_sub.png", "xyclus_x", "xyclus_y", 12, [[-12.8, 12.8], [-12.8, 12.8]]),
            ("pixmapglob_sub.png", "pixmapglob_x", "pixmapglob_y", 8, [[-0.5, 7.5], [-0.5, 7.5]])
        ]

        for out_name, kx, ky, nbins, rng in plot_defs:
            if kx in sc and ky in sc:
                x = sc[kx]
                y = sc[ky]
                w = sc.get("pixmapglob_w") if "pixmap" in out_name else None
                
                # Filter valid values
                if w is None:
                    mask = (x > -100) & (y > -100) 
                    x = x[mask]; y = y[mask]
                
                if len(x) > 0:
                    plt.figure(figsize=(8, 6))
                    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=rng, weights=w)
                    
                    plt.imshow(H.T, origin='lower', extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]], 
                               cmap='plasma', aspect='auto')
                    
                    plt.colorbar(label='Counts')
                    plt.title(out_name.replace(".png", ""))
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.savefig(os.path.join(dirs["plots"], out_name))
                    plt.close()

    print(f" Legacy Plots saved in: {dirs['plots']}")