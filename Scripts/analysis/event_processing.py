import numpy as np
import uproot
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, Optional
from analysis.io_manager import get_project_root

# CONSTANTS CONFIGURATION
PEDESTAL = 50
THR = 75
PIX_SIZE = 3.2
THR_CENTROID = 50

# Channel Maps
CH_ID_2_MAP_X = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
                          0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7], dtype=int)
CH_ID_2_MAP_Y = np.array([6,5,7,4,7,4,6,5,6,5,7,4,7,4,6,5,6,5,7,4,7,4,6,5,6,5,7,4,7,4,6,5,
                          2,1,3,0,3,0,2,1,2,1,3,0,3,0,2,1,2,1,3,0,3,0,2,1,2,1,3,0,3,0,2,1], dtype=int)

def load_calibration_dynamic(run_dir):
    corr = np.ones(64, dtype=float)
    calib_source = "None (Unity)"
    expected_path = os.path.join(run_dir, "calibration.xlsx")
    
    if os.path.exists(expected_path):
        try:
            df_corr = pd.read_excel(expected_path, sheet_name='Correzioni', dtype=str)
            count = 0
            for _, row in df_corr.iterrows():
                try:
                    ch = int(float(row["Channel"]))
                    val = float(str(row["Correction"]).replace(',', '.'))
                    if 0 <= ch < 64: 
                        corr[ch] = val
                        count += 1
                except: continue
            calib_source = f"File: {os.path.basename(expected_path)} ({count} ch)"
        except Exception as e:
            print(f"    Error reading calibration: {e}")
            calib_source = "Error (Unity)"
    
    return corr, calib_source

def elapsed_time(trigTime):
    if len(trigTime) == 0: return 1.0
    return (np.max(trigTime) - np.min(trigTime)) / (10**6)

# Functions for background subtraction and plotting

def plot_subtraction_check(bins, raw_c, bkg_scaled_c, net_c, name, run_path, scale_factor):
    """Generates the pre/post subtraction comparison plot in the Results folder."""
    
    # Determine the run and date from the .root file
    # Example run_path: .../Data_Converted/27_11_2025/Run1.root
    run_file = os.path.basename(run_path)                  # "Run1.root"
    run_name = os.path.splitext(run_file)[0]               # "Run1"
    date_str = os.path.basename(os.path.dirname(run_path)) # "27_11_2025"
    
    # Build the correct path to Results
    root_dir = get_project_root()
    out_dir = os.path.join(root_dir, "Results", date_str, run_name, "bkg_checks")
    os.makedirs(out_dir, exist_ok=True)
    
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Raw Data
    ax.step(centers, raw_c, where='mid', color='gray', alpha=0.5, label='Raw Data', linewidth=1)
    # Scaled Background
    ax.step(centers, bkg_scaled_c, where='mid', color='red', alpha=0.6, linestyle='--', label=f'Scaled Bkg (factor={scale_factor:.3f})')
    # Net Data
    ax.step(centers, net_c, where='mid', color='blue', alpha=0.8, label='Net Spectrum', linewidth=1.5)
    
    ax.set_title(f"Background Subtraction: {name}\nRun: {run_name}")
    ax.set_xlabel("ADC Channel")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.grid(alpha=0.3)
    
    max_y = np.max(raw_c)
    if max_y > 0:
        ax.set_ylim(bottom=0.1, top=max_y * 1.1) 
        
    out_path = os.path.join(out_dir, f"subtraction_{name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

def apply_subtraction_and_plot(res_run, bkg_data, run_path):
    """Performs subtraction and calls the plot function for the main spectra."""
    t_run = res_run.get("acquisition_time_sec", 1.0)
    t_bkg = bkg_data.get("acquisition_time_sec", 1.0)
    if t_bkg <= 0: t_bkg = 1.0
    
    scale_factor = t_run / t_bkg
    print(f"    Scaling Factor: {scale_factor:.4f}")
    
    res_sub = res_run.copy()
    
    # Histograms to subtract (add here if others are needed)
    targets = ["total_charge", "main_clust_charge", "main_pix_charge"]
    
    for key in targets:
        if key in res_run["histograms"] and key in bkg_data["histograms"]:
            # Extract bins and counts for the Run
            bins_run, counts_run = res_run["histograms"][key]
            
            # Extract bins and counts for the Bkg
            bins_bkg, counts_bkg = bkg_data["histograms"][key]
            
            # If the binning is the same, proceed
            if len(bins_run) == len(bins_bkg) and np.allclose(bins_run, bins_bkg):
                # Scale the background
                counts_bkg_scaled = counts_bkg * scale_factor
                
                # Subtraction
                counts_net = counts_run - counts_bkg_scaled
                
                # Verification plot
                plot_subtraction_check(bins_run, counts_run, counts_bkg_scaled, counts_net, key, run_path, scale_factor)
                
                # Update the results dictionary with net data
                res_sub["histograms"][key] = (bins_run, counts_net)
            else:
                print(f"    Binning mismatch for {key}. Cannot subtract.")
                
    return res_sub


def collect_all_histograms(run_path: str, bkg_data: Optional[Dict] = None) -> Dict[str, Any]:
    print(f"Processing: {os.path.basename(run_path)}")
    
    run_dir = os.path.dirname(run_path)
    corr, calib_info = load_calibration_dynamic(run_dir)
    print(f"    Calibration: {calib_info}")

    try:
        with uproot.open(run_path) as f:
            data = f["fersTree"].arrays(["trigTime", "channelID", "channelDataLG"], library="np")
    except: return {}

    trigTime = data["trigTime"]
    channelID = data["channelID"]
    channelData = data["channelDataLG"]
    
    total_charge = []
    main_clust_charge = []
    main_pix_charge = [] 
    
    xglob, yglob = [], []
    xclus, yclus = [], []
    
    xyglob_x, xyglob_y = [], []
    xyclus_x, xyclus_y = [], []
    
    pixmapglob_x, pixmapglob_y, pixmapglob_w = [], [], []
    
    npix5by5_list = []
    npix7by7_list = []
    npixdiff7vs5_list = [] 
    
    pixmapglob = np.zeros((8, 8), dtype=float)
    ampliDistLG = np.zeros((64, 4096), dtype=int)

    for j in tqdm(range(len(channelID)), desc="Events", leave=False):
        chids = channelID[j]
        vals = channelData[j]
        if len(chids) == 0: continue

        vals_corr = vals * corr[chids]
        max_i = int(np.argmax(vals_corr))
        max_id = chids[max_i]
        max_v_corr = vals_corr[max_i]

        sum_signal = 0
        sum_cluster = 0
        n5 = 0; n7 = 0
        
        chglob_centroid = 0.0; pix_x = 0.0; pix_y = 0.0
        chclus = 0.0; poscx = 0.0; poscy = 0.0

        for k, cid in enumerate(chids):
            v_raw = vals[k] 
            v_c = vals_corr[k]
            adc = v_c - PEDESTAL
            
            if v_c > THR_CENTROID:
                chglob_centroid += adc
                pix_x += adc * CH_ID_2_MAP_X[cid]
                pix_y += adc * CH_ID_2_MAP_Y[cid]

            if v_c < THR: continue
            
            sum_signal += adc
            
            if 0 <= adc < 4096: 
                ampliDistLG[cid, int(adc)] += 1
            
            weight = v_c / 4095.0
            pixmapglob[CH_ID_2_MAP_X[cid], CH_ID_2_MAP_Y[cid]] += weight
            
            pixmapglob_x.append(CH_ID_2_MAP_X[cid])
            pixmapglob_y.append(CH_ID_2_MAP_Y[cid])
            pixmapglob_w.append(weight)

            dx = abs(CH_ID_2_MAP_X[cid] - CH_ID_2_MAP_X[max_id])
            dy = abs(CH_ID_2_MAP_Y[cid] - CH_ID_2_MAP_Y[max_id])
            
            if dx < 3 and dy < 3:
                sum_cluster += adc
                n5 += 1
                chclus += adc
                poscx += adc * CH_ID_2_MAP_X[cid]
                poscy += adc * CH_ID_2_MAP_Y[cid]
            
            if dx < 4 and dy < 4:
                n7 += 1

        if max_v_corr < 4000:
            total_charge.append(sum_signal)
            if sum_cluster > 0:
                main_clust_charge.append(sum_cluster)
            
            main_pix_charge.append(int(vals[max_i] - PEDESTAL))
            
            if chglob_centroid > 0:
                xg = ((pix_x / chglob_centroid) - 3.5) * PIX_SIZE
                yg = ((pix_y / chglob_centroid) - 3.5) * PIX_SIZE
                xglob.append(xg); yglob.append(yg)
                xyglob_x.append(xg); xyglob_y.append(yg)

            if chclus > 0:
                xc = ((poscx / chclus) - 3.5) * PIX_SIZE
                yc = ((poscy / chclus) - 3.5) * PIX_SIZE
                xclus.append(xc); yclus.append(yc)
                xyclus_x.append(xc); xyclus_y.append(yc)

            npix5by5_list.append(n5)
            npix7by7_list.append(n7)
            npixdiff7vs5_list.append(n7 - n5)

    res = {"histograms": {}, "maps": {}, "raw_data": {}, "scatter_data": {}}
    res["calibration_info"] = calib_info 
    
    def add_hist(name, data, bins, rng):
        cnt, bns = np.histogram(data, bins=bins, range=rng)
        res["histograms"][name] = (bns, cnt)

    add_hist("total_charge", total_charge, 251, (-5, 10005))
    add_hist("main_clust_charge", main_clust_charge, 351, (-5, 7005))
    add_hist("main_pix_charge", main_pix_charge, 351, (-5, 7005))
    
    add_hist("npix5by5", npix5by5_list, 26, (0, 25))
    add_hist("npix7by7", npix7by7_list, 50, (0, 49))
    add_hist("npixdiff7vs5", npixdiff7vs5_list, 26, (0, 25))

    add_hist("xglob", xglob, 12, (-12.8, 12.8))
    add_hist("yglob", yglob, 12, (-12.8, 12.8))
    add_hist("xclus", xclus, 12, (-12.8, 12.8))
    add_hist("yclus", yclus, 12, (-12.8, 12.8))

    res["scatter_data"]["xyglob_x"] = np.array(xyglob_x)
    res["scatter_data"]["xyglob_y"] = np.array(xyglob_y)
    res["scatter_data"]["xyclus_x"] = np.array(xyclus_x)
    res["scatter_data"]["xyclus_y"] = np.array(xyclus_y)
    
    res["scatter_data"]["pixmapglob_x"] = np.array(pixmapglob_x)
    res["scatter_data"]["pixmapglob_y"] = np.array(pixmapglob_y)
    res["scatter_data"]["pixmapglob_w"] = np.array(pixmapglob_w)

    res["maps"]["pixmapglob"] = pixmapglob.T 
    res["ampliDistLG"] = ampliDistLG
    res["acquisition_time_sec"] = elapsed_time(trigTime)
    
    res["raw_data"]["total_charge"] = np.array(total_charge)

    if bkg_data:
        # Use the new internal function to subtract AND generate plots
        res_sub = apply_subtraction_and_plot(res, bkg_data, run_path)
        res_sub["scatter_data"] = res["scatter_data"] 
        return res_sub
    
    return res