#!/usr/bin/env python3
import os
import argparse
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
from lmfit.models import ConstantModel
from lmfit.models import GaussianModel, PolynomialModel
from lmfit.models import Gaussian2dModel
from math import sqrt, pi
from tqdm import tqdm
import csv
from matplotlib.ticker import MultipleLocator, MaxNLocator
import pandas as pd
from uncertainties import ufloat


# === add after imports ===
def _fmt_eu(val):
    """Return value as string with decimal comma (e.g. 3,14)."""
    if val is None:
        return ""
    s = str(val)
    return s.replace('.', ',')

def _csv_writer_semicolon(path):
    """CSV writer with ';' delimiter to coexist with decimal commas."""
    return open(path, 'w', newline=''), csv.writer(open(path, 'w', newline=''), delimiter=';')



# --- Gaussian‐fit helpers --------------------------------------------

def move_mean(x, w):
    return uniform_filter1d(x, size=w, mode='nearest')

def _estimate_peakpos_from_lratio(bins, xcounts, peaks, widths):
    """Return (start, stop) intervals for each peak index & width."""
    limits = []
    for pk, w in zip(peaks, widths):
        i0 = max(0, int(pk - w))
        i1 = min(len(bins)-1, int(pk + w))
        limits.append((bins[i0], bins[i1]))
    return np.array(limits)

def _line_fitter(x, y, limits):
    start, stop = limits
    i_start = np.where(x > start)[0][0]
    i_stop  = np.where(x < stop)[0][-1]

    # edges → centers
    x_fit = (x[i_start:i_stop+1][1:] + x[i_start:i_stop+1][:-1]) / 2
    y_fit = y[i_start:i_stop]
    weights = np.sqrt(y_fit)

    mod  = GaussianModel()
    pars = mod.guess(y_fit, x=x_fit)
    result = mod.fit(y_fit, pars, x=x_fit, weights=weights)

    return result


def _fit_peaks(x, y, limits):
    """
    Fit each (x,y) slice defined by limits with a Gaussian.
    Returns arrays: centers, center_errs, fwhms, fwhm_errs, amps (area), amp_errs.
    """
    n = len(limits)
    centers = np.zeros(n);  errs_c  = np.zeros(n)
    fwhms   = np.zeros(n);  errs_f  = np.zeros(n)
    amps    = np.zeros(n);  errs_a  = np.zeros(n)

    for i, lim in enumerate(limits):
        res = _line_fitter(x, y, lim)
        p   = res.params

        # Extract
        ctr     = p['center'].value
        ctr_err = p['center'].stderr
        sigma   = p['sigma'].value
        dsig    = p['sigma'].stderr

        amp     = p['amplitude'].value
        amp_err = p['amplitude'].stderr

        factor = 2 * np.sqrt(2 * np.log(2))

        # If we have a covariance matrix, read var(σ) from it.
        if res.covar is not None and hasattr(res, 'var_names') and ('sigma' in res.var_names):
            try:
                j = res.var_names.index('sigma')
                var_sigma = res.covar[j, j]
                if np.isfinite(var_sigma) and var_sigma > 0:
                    dsig = np.sqrt(var_sigma)
            except Exception:
                pass  # keep dsig as-is (stderr or None)

        # --- Robust fallback only if covariance/std.err unavailable
        if dsig is None or not np.isfinite(dsig) or dsig <= 0:
            residuals = res.residual
            noise = np.std(residuals)
            height = amp / (sigma * np.sqrt(2 * np.pi))
            est_dsig = noise / (max(height, 1e-12) * factor)  # safeguard
            dsig = est_dsig

        # Propagate with 'uncertainties' to get FWHM and its σ
        fwhm_u = ufloat(sigma, dsig) * factor
        fwhm   = fwhm_u.nominal_value
        fwhm_err = fwhm_u.std_dev

        # Store
        centers[i] = ctr
        errs_c[i]  = ctr_err if ctr_err is not None else np.nan
        fwhms[i]   = fwhm
        errs_f[i]  = fwhm_err
        amps[i]    = amp
        errs_a[i]  = amp_err if amp_err is not None else np.nan

    return centers, errs_c, fwhms, errs_f, amps, errs_a

def background(x, b0, b1, b2):
        """2nd-order polynomial background"""
        return b0 + b1*x + b2*x**2



def _find_peaks_2d(Z, min_distance=3, threshold_abs= None , max_peaks=2):
    """
    Return up to `max_peaks` local maxima of Z as a list of (ix, iy, value),
    sorted by value descending. `min_distance` is in bins.
    No implicit percentile threshold is applied.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.size == 0 or not np.isfinite(Z).any():
        return []

    # light smoothing to reduce pixel noise (not background subtraction)
    Zs = gaussian_filter(Z, sigma= 1)

    neigh = maximum_filter(Zs, size=(2*min_distance+1))
    if threshold_abs is None:
        mask = (Zs == neigh)
    else:
        mask = (Zs == neigh) & (Zs >= float(threshold_abs))

    lbl, nlab = label(mask)
    peaks = []
    for k in range(1, nlab+1):
        coords = np.argwhere(lbl == k)
        if coords.size == 0:
            continue
        vals = Zs[coords[:, 0], coords[:, 1]]
        i = int(np.argmax(vals))
        ix, iy = coords[i]
        peaks.append((int(ix), int(iy), float(Zs[ix, iy])))

    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks[:max_peaks]


# --- ROOT event‐loop histogram collector ----------------------------

def collect_all_histograms(root_file):
    """
    Reads `fersTree` from the given ROOT file and returns a dict containing:
      - ampliDistLG (64×4096 array)
      - total_charge, main_clust_charge, main_pix_charge lists
      - xyglob_x, xyglob_y, xyclus_x, xyclus_y, xyglobloose_x, ...
      - pixmapglob (8×8), pixmapclus (8×8)
      - npix5by5_list, npix7by7_list, npixdiff7vs5_list
    """
    with uproot.open(root_file) as f:
        tree = f["fersTree"]
        data = tree.arrays(["channelID","channelDataLG"], library="np")
    channelID    = data["channelID"]
    channelData  = data["channelDataLG"]

    # C++ constants
    ped      = 50
    thr      = 95
    pix_size = 3.2
    chID2mapX = np.array([
        0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
        0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7
    ],dtype=int)
    chID2mapY = np.array([
        6,5,7,4,7,4,6,5,6,5,7,4,7,4,6,5,6,5,7,4,7,4,6,5,6,5,7,4,7,4,6,5,
        2,1,3,0,3,0,2,1,2,1,3,0,3,0,2,1,2,1,3,0,3,0,2,1,2,1,3,0,3,0,2,1
    ],dtype=int)

    # allocate
    ampliDistLG = np.zeros((64,4096),dtype=int)
    total_charge, main_clust_charge, main_pix_charge = [], [], []
    xglob, yglob = [], []
    xclus, yclus = [], []
    xyglob_x, xyglob_y = [], []
    xyclus_x, xyclus_y = [], []
    xyglobloose_x, xyglobloose_y = [], []
    xyclusloose_x, xyclusloose_y = [], []
   
    pixmapglob = np.zeros((8,8),dtype=float)
    pixmapglob_x, pixmapglob_y, pixmapglob_w = [], [], []
    pixmapclus = np.zeros((8,8),dtype=float)
    npix5by5_list, npix7by7_list, npixdiff7vs5_list = [], [], []

    nentries = len(channelID)
    for j in tqdm(range(nentries), desc=f'Reading {os.path.basename(root_file)}'):
        chids = channelID[j]
        vals  = channelData[j]
        if len(chids)==0:
            continue
        max_i = int(np.argmax(vals))
        max_v = int(vals[max_i])

        sum_signal = 0
        sum_cluster= 0
        n5 = n7 = 0
        chglob=chclus=0
        pix_x=pix_y=0
        poscx=poscy=0

        for cid, v in zip(chids, vals):
            if v < thr: continue
            adc = v - ped
            sum_signal += adc
            if 0 <= adc < 4095:
                ampliDistLG[cid, int(adc)] += 1
            pixmapglob[chID2mapX[cid],chID2mapY[cid]] += v/4095.0
            chglob += adc
            pix_x += adc * chID2mapX[cid]
            pix_y += adc * chID2mapY[cid]
            pix_w = float(v/4095.0)  # use same weight as the 8x8 map
            pixmapglob_x.append(int(chID2mapX[cid]))   # ← index, not sum
            pixmapglob_y.append(int(chID2mapY[cid]))
            pixmapglob_w.append(pix_w)

            

            # 5×5 cluster
            if (abs(chID2mapX[cid]-chID2mapX[chids[max_i]])<3
             and abs(chID2mapY[cid]-chID2mapY[chids[max_i]])<3):
                sum_cluster += adc
                n5 += 1
                pixmapclus[chID2mapX[cid],chID2mapY[cid]] += v/4000.0
                chclus += adc
                poscx  += adc * chID2mapX[cid]
                poscy  += adc * chID2mapY[cid]
            # 7×7 cluster
            if (abs(chID2mapX[cid]-chID2mapX[chids[max_i]])<4
             and abs(chID2mapY[cid]-chID2mapY[chids[max_i]])<4):
                n7 += 1

        if max_v < 4000:
            total_charge.append(sum_signal)
            if sum_cluster>0:
                main_clust_charge.append(sum_cluster)
            main_pix_charge.append(int(vals[max_i]-ped))

            xg = ((pix_x/chglob)-3.5)*pix_size
            yg = ((pix_y/chglob)-3.5)*pix_size
            xglob.append(xg); yglob.append(yg)
            xyglob_x.append(xg); xyglob_y.append(yg)

            xc = ((poscx/chclus)-3.5)*pix_size
            yc = ((poscy/chclus)-3.5)*pix_size
            xclus.append(xc); yclus.append(yc)
            xyclus_x.append(xc); xyclus_y.append(yc)

            xyglobloose_x.append(xg); xyglobloose_y.append(yg)
            xyclusloose_x.append(xc); xyclusloose_y.append(yc)

            npix5by5_list.append(n5)
            npix7by7_list.append(n7)
            npixdiff7vs5_list.append(n7-n5)

    return {
        "ampliDistLG": ampliDistLG,
        "total_charge": np.array(total_charge),
        "main_clust_charge": np.array(main_clust_charge),
        "main_pix_charge": np.array(main_pix_charge),
        "xglob": np.array(xglob), 
        "yglob": np.array(yglob),
        "xclus": np.array(xclus),
        "yclus": np.array(yclus),
        "xyglob_x": np.array(xyglob_x),"xyglob_y": np.array(xyglob_y),
        "xyclus_x": np.array(xyclus_x),"xyclus_y": np.array(xyclus_y),
        "xyglobloose_x": np.array(xyglobloose_x),
        "xyglobloose_y": np.array(xyglobloose_y),
        "xyclusloose_x": np.array(xyclusloose_x),
        "xyclusloose_y": np.array(xyclusloose_y),
        "pixmapglob": pixmapglob,
        "pixmapglob_x": np.array(pixmapglob_x),
        "pixmapglob_y": np.array(pixmapglob_y),
        "pixmapglob_w": np.array(pixmapglob_w),
        "pixmapclus": pixmapclus,
        "npix5by5": np.array(npix5by5_list),
        "npix7by7": np.array(npix7by7_list),
        "npixdiff7vs5": np.array(npixdiff7vs5_list)
    }

# --- Main subtraction & plotting ------------------------------------

def process_spectrum_minus_background(spec_file, spec_time,
                                      bkg_file,  bkg_time,
                                      fit_mode: int = 4,
                                      bins1d_pos: int = 12,
                                      bins2d: int = 12,
                                      pixbins: int = 8):
    """
    fit_mode:
      1 = Only histograms (no fitting)
      2 = Fit only total_charge
      3 = Fit total_charge, xglob, yglob; 2D only histogram
      4 = Full analysis (all fits and histograms)
    """
    spec = collect_all_histograms(spec_file)
    bkg  = collect_all_histograms(bkg_file)
    scale = spec_time / bkg_time
    # [Ref original structure]:contentReference[oaicite:0]{index=0}

    base = os.path.splitext(os.path.basename(spec_file))[0]
    out_dir = f"{base}_bkg_sub"
    os.makedirs(out_dir, exist_ok=True)

    # --- channel amplitude distributions ---
    ampspec = spec["ampliDistLG"]
    ampbkg  = bkg["ampliDistLG"] * scale
    ampdiff = ampspec - ampbkg
    ampdiff[ampdiff<0] = 0

    # --- 1D summary histograms ---
    names_1d=[
        'total_charge','main_clust_charge','main_pix_charge',
        'npix5by5','npix7by7','npixdiff7vs5',
        'xglob','yglob','xclus','yclus'
     ]  #:contentReference[oaicite:1]{index=1}
 
    bins_1d_defs = {
        "total_charge":    (251, (-5, 10005)),
        "main_clust_charge":(351,(-5,7005)),
        "main_pix_charge": (351,(-5,7005)),
        "npix5by5":        (26,(0,25)),
        "npix7by7":        (50,(0,49)),
        "npixdiff7vs5":    (26,(0,25)),
        "xglob":           (bins1d_pos,    (-12.8, 12.8)),
        "yglob":           (bins1d_pos,    (-12.8, 12.8)),
        "xclus":           (bins1d_pos,    (-12.8, 12.8)),
        "yclus":           (bins1d_pos,    (-12.8, 12.8)),
    }

    # precompute bkg hist counts
    bkg_hists = {}
    raw_hists = {}
    for n in names_1d:
        nb, rg = bins_1d_defs[n]
        # raw spectrum
        h_raw, _ = np.histogram(spec[n], bins=nb, range=rg)
        raw_hists[n] = h_raw
        # scaled background
        h_bkg, _ = np.histogram(bkg[n], bins=nb, range=rg)
        bkg_hists[n] = h_bkg * scale

    for name in tqdm(names_1d, desc='1D hist sub'):
        nb, rg = bins_1d_defs[name]
        ed = np.linspace(rg[0], rg[1], nb+1)
        h_spec = raw_hists[name]
        h_bkg = bkg_hists[name]
        hd = h_spec - h_bkg
        hd[hd < 0] = 0

        # Always draw the subtracted histogram for non-total_charge here
        if name != 'total_charge':
            # If xglob/yglob and we’re NOT fitting (modes 1 or 2), just make the histogram
            if name in ('xglob', 'yglob') and fit_mode not in (3, 4):
                fig, ax = plt.subplots()
                ax.hist(ed[:-1], bins=ed, weights=hd, histtype='stepfilled', alpha=0.6)
                ax.set_title(f"{name}_sub")
                ax.set_xlabel(name)
                ax.set_ylabel("Counts")
                if name in ('xglob','yglob','xclus','yclus'):
                    ax.set_xlim(-12.8, 12.8)
                fig.savefig(os.path.join(out_dir, f"{name}_sub.png"))
                plt.close(fig)
                continue

            # If xglob/yglob and we ARE fitting (modes 3 or 4), run the original fit branch
            if name in ('xglob', 'yglob') and fit_mode in (3, 4):
                # smooth & find peaks
                mm = move_mean(hd, 5)
                peaks, info = find_peaks(mm, prominence=10, width=2)
                if len(peaks) == 0:
                    raise RuntimeError(f"No {name} peaks after subtraction")

                # build (low, high) limits for each peak
                limits = _estimate_peakpos_from_lratio(ed, hd, peaks, info['widths'])
                # fit each slice with our Gaussian helper
                centers, center_errs, fwhms, fwhm_errs, amps, amp_errs = _fit_peaks(ed, hd, limits)

                # write out CSV of results
                csv_path = os.path.join(out_dir, f'{name}_fit_sub_fit.csv')
                with open(csv_path, 'w', newline='') as fh:
                    w = csv.writer(fh, delimiter=';')
                    w.writerow([
                        "Amplitude", "Amplitude Error",
                        "Center",    "Center Error",
                        "FWHM",      "FWHM Error",
                        "Limit_Low", "Limit_High"
                    ])
                    for amp, amp_err, ctr, ctr_err, fwhm, fwhm_err, (low, high) in zip(
                        amps, amp_errs, centers, center_errs, fwhms, fwhm_errs, limits
                    ):
                        w.writerow([
                            _fmt_eu(amp), _fmt_eu(amp_err),
                            _fmt_eu(ctr), _fmt_eu(ctr_err),
                            _fmt_eu(fwhm), _fmt_eu(fwhm_err),
                            _fmt_eu(low), _fmt_eu(high)
                        ])

                # now plot the histogram + Gaussian fits
                xcenters = (ed[:-1] + ed[1:]) / 2
                fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
                ax.step(xcenters, hd, where='mid',label=f'{name}_sub', color='black')
                for ((amp, ctr, fwhm), (low, high)) in zip(zip(amps, centers, fwhms), limits):
                    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                    height = amp / (sigma * np.sqrt(2 * np.pi))
                    xs_smooth = np.linspace(low, high, 500)
                    ys_smooth = height * np.exp(-0.5 * ((xs_smooth - ctr) / sigma)**2)
                    ax.axvline(ctr, linestyle=':', color='#6e6e33')
                    ax.axvspan(low, high, color="#6e6e33", alpha=0.1)
                    ax.fill_between(xs_smooth, ys_smooth, alpha=0.3)
                    ax.plot(xs_smooth, ys_smooth, '--', label=f"Fit Center (ADU)")
                ax.set_title(f"{'Projected Counts along X-Axis' if name=='xglob' else 'Projected Counts along Y-Axis'} ")
                ax.set_xlabel("Position (mm)"); ax.set_xlim(-12.8, 12.8)
                ax.locator_params(axis='x', nbins=15)
                ax.set_ylabel('Counts')
                ax.xaxis.set_major_locator(MultipleLocator(1.8))
                ax.legend()
                fig.savefig(os.path.join(out_dir, f'{name}_fit_sub.png'), dpi=200, bbox_inches="tight")
                plt.close(fig)
                continue

            # default plot (xclus/yclus/npix*/etc.)
            fig, ax = plt.subplots()
            ax.hist(ed[:-1], bins=ed, weights=hd, histtype='stepfilled', alpha=0.6)
            ax.set_title(f"{name}_sub")
            ax.set_xlabel(name)
            if name in ('xglob','yglob','xclus','yclus'):
                ax.set_xlim(-12.8, 12.8)
            ax.set_ylabel("Counts")
            fig.savefig(os.path.join(out_dir, f"{name}_sub.png"))
            plt.close(fig)
            continue

        # === total_charge branch ===
        if name == 'total_charge':
            # combined plot: raw, background, and subtracted (always)
            fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
            ax.hist(ed[:-1], bins=ed, weights=h_spec, histtype='step', color='red', label='Raw Histogram')
            ax.hist(ed[:-1], bins=ed, weights=h_bkg,  histtype='step', color='yellow', label='Background Histogram (scaled)')
            ax.hist(ed[:-1], bins=ed, weights=hd,     histtype='stepfilled', alpha=0.7, label='Spectrum - Background')
            ax.set_title("Energy Spectrum (background subtracted)")
            ax.set_xlabel('ADU'); ax.set_ylabel('Counts'); ax.legend()
            fig.savefig(os.path.join(out_dir, f"{name}_combined.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

            # Only fit total_charge when modes 2,3,4
            if fit_mode in (2, 3, 4):
                # --- Gaussian fit on the FULL subtracted histogram (unchanged) ---
                xcenters = (ed[:-1] + ed[1:]) / 2
                y_fit    = hd.copy()
                weights  = np.sqrt(np.clip(y_fit, 1, None))

                gmod  = GaussianModel()
                pars  = gmod.guess(y_fit, x=xcenters)
                result = gmod.fit(y_fit, pars, x=xcenters, weights=weights)

                # --- choose peak window as before ---
                mm = move_mean(hd, 5)
                peaks, info = find_peaks(mm, prominence=10, width=2)
                if len(peaks) == 0:
                    raise RuntimeError("no peaks after subtraction")
                idx_best = np.argmax(mm[peaks])      # tallest peak
                best_pk  = int(peaks[idx_best])
                w        = float(info['widths'][idx_best])

                # map center-index window -> edges
                i0 = max(0,           int(round(best_pk - w/2)))
                i1 = min(len(ed) - 2, int(round(best_pk + w/2)))
                low, high = ed[i0], ed[i1 + 1]

                # --- local refit on [low, high] like 03_07 ---
                limits = [(low, high)]
                centers, center_errs, fwhms, fwhm_errs, amps, amp_errs = _fit_peaks(ed, hd, limits)
                ctr = centers[0]

                # --- draw ---
                xcenters = 0.5*(ed[:-1] + ed[1:])
                fig, ax = plt.subplots(figsize=(12,7), constrained_layout=True)
                ax.step(xcenters, hd, where='mid', color='black', label='Energy Spectrum')

                for ctr_i, amp, fwhm in zip(centers, amps, fwhms):
                    sigma   = fwhm / (2*np.sqrt(2*np.log(2)))
                    height  = amp / (sigma*np.sqrt(2*np.pi))  # area -> peak height
                    xs      = np.linspace(low, high, 800)
                    ys      = height * np.exp(-0.5*((xs - ctr_i)/sigma)**2)
                    ax.plot(xs, ys, '-', color='#d4d430', label='Gaussian fit')
                    ax.axvline(ctr_i, linestyle=':', color='#676727')
                    ax.axvspan(low, high, color='#d4d430', alpha=0.1)

                ax.set_title('Energy Spectrum (Gaussian Fit)')
                ax.set_xlabel('ADU'); ax.set_ylabel('Counts'); ax.legend()
                fig.savefig(os.path.join(out_dir, f"{name}_fit_sub.png"), dpi=200, bbox_inches="tight")
                plt.close(fig)

               # after drawing the fitted curve for total_charge...
                factor = 2 * np.sqrt(2 * np.log(2))  # 2.35482...
                bin_width = ed[1] - ed[0]

                # take values from the LOCAL refit (centers[0], fwhms[0], etc.)
                ctr_local      = centers[0]
                ctr_err_local  = center_errs[0]
                fwhm_local     = fwhms[0]                 # already in ADU
                fwhm_err_local = fwhm_errs[0]
                sigma_local    = fwhm_local / factor
                sigma_err_loc  = fwhm_err_local / factor if np.isfinite(fwhm_err_local) else np.nan

                # OPTIONAL: also record FWHM in "bins" so it’s obvious if someone compares by eye
                fwhm_bins = fwhm_local / bin_width
                fwhm_bins_err = fwhm_err_local / bin_width if np.isfinite(fwhm_err_local) else np.nan

                csv_path = os.path.join(out_dir, f'{name}_gauss_fit.csv')
                with open(csv_path, 'w', newline='') as fh:
                    wcsv = csv.writer(fh, delimiter=';')
                    wcsv.writerow([
                        "Gaussian_Amplitude","Amplitude_Error",
                        "Center_ADU","Center_Error_ADU",
                        "Sigma_ADU","Sigma_Error_ADU",
                        "FWHM_ADU","FWHM_Error_ADU",
                        "FWHM_bins","FWHM_bins_Error",
                        "Limit_Low_ADU","Limit_High_ADU"
                    ])
                    # amplitude from local refit area parameter (amps[0])
                    amp_local     = amps[0]
                    amp_err_local = amp_errs[0]
                    wcsv.writerow([
                        _fmt_eu(amp_local), _fmt_eu(amp_err_local),
                        _fmt_eu(ctr_local), _fmt_eu(ctr_err_local),
                        _fmt_eu(sigma_local), _fmt_eu(sigma_err_loc),
                        _fmt_eu(fwhm_local), _fmt_eu(fwhm_err_local),
                        _fmt_eu(fwhm_bins), _fmt_eu(fwhm_bins_err),
                        _fmt_eu(low), _fmt_eu(high)
                    ])


        # default behavior for other histograms (safety fallback)
        fig, ax = plt.subplots()
        ax.hist(ed[:-1], bins=ed, weights=hd,
                histtype='stepfilled', alpha=0.6, label='spec - bkg')
        ax.set_title(f"{name}_sub")
        ax.set_xlabel(name)
        if name in ('xglob','yglob','xclus','yclus'):
            ax.set_xlim(-12.8, 12.8)
        ax.set_ylabel("Counts")
        ax.legend()
        fig.savefig(os.path.join(out_dir, f"{name}_sub.png"))
        plt.close(fig)

    # --- 2D histograms (xyglob, xyclus, xyglobloose, xyclusloose) ---
    hist2d_defs = {
        "xyglob":        (spec["xyglob_x"], spec["xyglob_y"],     bins2d,   "mm"),
        "xyclus":        (spec["xyclus_x"], spec["xyclus_y"],     bins2d,   "mm"),
        "xyglobloose":   (spec["xyglobloose_x"], spec["xyglobloose_y"],  bins2d,   "mm"),
        "xyclusloose":   (spec["xyclusloose_x"], spec["xyclusloose_y"],  bins2d,   "mm"),
        "pixmapglob":   (spec["pixmapglob_x"],   spec["pixmapglob_y"],  pixbins,  "pix"),   }
    for name, (x_s, y_s, bins, unit) in hist2d_defs.items():
        # spec
        Hs,xe,ye = np.histogram2d(x_s, y_s, bins=bins2d, range=[[-12.8,12.8],[-12.8,12.8]])
        # bkg
        x_b,y_b,_ = (bkg[name+"_x"], bkg[name+"_y"], bins2d)
        Hb,_ , _  = np.histogram2d(x_b, y_b, bins=bins2d, range=[[-12.8,12.8],[-12.8,12.8]])
        Hd = Hs - Hb * scale
        Hd[Hd<0] = 0

       # ==== xyglob: 2D Gaussian fit only in mode 4 ====
        if name == 'xyglob' and fit_mode == 4:
            # ---- Enhanced 2D fit: auto 1 or 2 Gaussians, always draw contours ----

            # 1) Grid & arrays
            x_centers = (xe[:-1] + xe[1:]) / 2
            y_centers = (ye[:-1] + ye[1:]) / 2
            X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
            Z = Hd.astype(float)

            # If nothing to fit, draw heatmap and exit
            if not np.isfinite(Z).any() or np.nanmax(Z) <= 0:
                fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
                im = ax.imshow(
                    Z.T, origin='lower',
                    extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
                    cmap='plasma'
                )
                cbar = plt.colorbar(im, ax=ax); cbar.set_label('Counts', fontsize=20); cbar.ax.tick_params(labelsize=16)
                ax.set_title("Hitmap (no valid counts for Gaussian fit)", fontsize=20)
                ax.tick_params(axis='both', labelsize=14)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=17))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=17))
                ax.set_xlabel('X-Axis (mm)', fontsize=20); ax.set_ylabel('Y-Axis (mm)', fontsize=20)
                fig.savefig(os.path.join(out_dir, f"{name}_with_contours.png"), dpi=200, bbox_inches="tight")
                plt.close(fig)
            else:
                x_flat, y_flat, z_flat = X.ravel(), Y.ravel(), Z.ravel()

                # 2) Moments for seeds
                total = float(np.nansum(Z))
                x0_m  = float(np.nansum(X * Z) / total)
                y0_m  = float(np.nansum(Y * Z) / total)
                sx_m  = float(np.sqrt(np.nansum(((X - x0_m)**2) * Z) / total)) or 1.0
                sy_m  = float(np.sqrt(np.nansum(((Y - y0_m)**2) * Z) / total)) or 1.0

                # 3) Up to 2 peaks (no implicit background threshold)
                peaks = _find_peaks_2d(Z, min_distance=1, threshold_abs=None, max_peaks=2)

                def _add_gaussian(prefix, ix, iy, amp_guess):
                    g = Gaussian2dModel(prefix=prefix)
                    ix = int(np.clip(ix, 0, len(x_centers)-1))
                    iy = int(np.clip(iy, 0, len(y_centers)-1))
                    cx = float(x_centers[ix]); cy = float(y_centers[iy])
                    p = g.make_params(
                        amplitude = max(float(amp_guess), 1.0),
                        centerx   = cx,  centery = cy,
                        sigmax    = max(sx_m, 1.0),
                        sigmay    = max(sy_m, 1.0),
                    )
                    
                    # keep Gaussians axis-aligned and without offsets
                    off_key = prefix + 'offset'
                    if off_key in p: p[off_key].set(value=0.0, vary=False)
                    rot_key = prefix + 'rotation'
                    if rot_key in p: p[rot_key].set(value=0.0, vary=False)

                    p[prefix+'sigmax'].set(min=-12.8, max=12.8)
                    p[prefix+'sigmay'].set(min=-12.8, max=12.8)
                    p[prefix+'sigmax'].set(min=1, max=10.0)
                    p[prefix+'sigmay'].set(min=1, max=10.0)
                    p[prefix+'amplitude'].set(min=0.0)
                    return g, p

                if len(peaks) == 0:
                    # fallback: use COM as seed
                    ix = int(np.argmin(np.abs(x_centers - x0_m)))
                    iy = int(np.argmin(np.abs(y_centers - y0_m)))
                    val = Z[ix, iy] if Z.size else 1.0
                    g1, p1 = _add_gaussian('g1_', ix, iy, val)
                    model = g1
                    params = p1
                    n_comp = 1
                elif len(peaks) == 1:
                    ix, iy, val = peaks[0]
                    g1, p1 = _add_gaussian('g1_', ix, iy, val)
                    model = g1
                    params = p1
                    n_comp = 1
                else:
                    (ix1, iy1, v1), (ix2, iy2, v2) = peaks[:2]
                    g1, p1 = _add_gaussian('g1_', ix1, iy1, v1)
                    g2, p2 = _add_gaussian('g2_', ix2, iy2, v2)

                    # ---- Proximity constraint: g2 center = g1 center + (dx, dy) with bounded deltas
                    dmax = 10.0  # <= adjust how "near" the peaks must be (in mm)
                    # keep g1 inside a margin so g1 + dx stays in [-12, 12]
                    p1['g1_centerx'].set(min=-12.8 + dmax, max=12.8 - dmax)
                    p1['g1_centery'].set(min=-12.8 + dmax, max=12.8 - dmax)

                    # delta parameters with bounds: |dx|, |dy| <= dmax
                    from lmfit import Parameters
                    params = Parameters()
                    params.update(p1)
                    params.update(p2)

                    # initial deltas from the peak seeds (clipped to bounds)
                    dx0 = float(x_centers[ix2] - x_centers[ix1])
                    dy0 = float(y_centers[iy2] - y_centers[iy1])
                    dx0 = float(np.clip(dx0, -dmax, dmax))
                    dy0 = float(np.clip(dy0, -dmax, dmax))
                    params.add('dx', value=dx0, min=-dmax, max=dmax)
                    params.add('dy', value=dy0, min=-dmax, max=dmax)

                    # tie g2 centers to g1 + deltas (expressions override min/max on g2 centers)
                    params['g2_centerx'].set(expr='g1_centerx + dx')
                    params['g2_centery'].set(expr='g1_centery + dy')

                    model  = g1 + g2
                    n_comp = 2

                # 5) Fit (Poisson weights)
                w = np.sqrt(np.clip(z_flat, 1, None))
                result = model.fit(z_flat, params, x=x_flat, y=y_flat, weights=w)

                # 6) Smooth grid for contours
                nx = ny = 200
                x_smooth = np.linspace(xe[0], xe[-1], nx)
                y_smooth = np.linspace(ye[0], ye[-1], ny)
                Xs, Ys = np.meshgrid(x_smooth, y_smooth, indexing='ij')

                # 7) Plot heatmap
                fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
                im = ax.imshow(
                    Z.T, origin='lower',
                    extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
                    cmap='plasma'
                )
                cbar = plt.colorbar(im, ax=ax); cbar.set_label('Counts', fontsize=20); cbar.ax.tick_params(labelsize=16)



                # Always draw per-component σ-contours at the requested probabilities
                # probs = [0.68, 0.95, 0.99]
                # ns    = np.sqrt(-2*np.log(1 - np.array(probs)))   # Mahalanobis radii
                # ---- Mixture contours (replace the per-component contour loop with this) ----
                # Evaluate mixture (sum of all fitted Gaussians) on the smooth grid
                Z_mix = np.zeros_like(Xs, dtype=float)
                for k in range(1, n_comp+1):
                    Z_mix += np.asarray(result.eval_components(x=Xs, y=Ys)[f'g{k}_'], dtype=float).reshape(Xs.shape)

                # Helper: find level thresholds so that the super-level set encloses the desired probability mass
                def _iso_prob_levels(Z, probs, xs, ys):
                    Z = np.asarray(Z, dtype=float)
                    # cell area (assumes rectilinear grid)
                    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
                    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
                    area = dx * dy

                    vals = Z.ravel()
                    # guard against all-NaN or non-positive surfaces
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0 or np.nanmax(vals) <= 0:
                        return np.array([])

                    order = np.argsort(vals)[::-1]       # descending
                    sv = vals[order]
                    csum = np.cumsum(sv) * area
                    total = csum[-1]
                    if total <= 0:
                        return np.array([])

                    mass = csum / total
                    levels = []
                    for p in probs:
                        j = np.searchsorted(mass, p)
                        j = min(j, sv.size - 1)
                        levels.append(float(sv[j]))
                    return np.array(levels, dtype=float)

                # Desired enclosed probabilities
                probs = [0.68, 0.95, 0.99]
                levels_raw = _iso_prob_levels(Z_mix, probs, x_smooth, y_smooth)

                # Build (level, label) pairs, sanitize & sort for contour()
                pairs = []
                zmin = float(np.nanmin(Z_mix))
                zmax = float(np.nanmax(Z_mix))
                for lvl, p in zip(levels_raw, probs):
                    if np.isfinite(lvl) and (zmin < lvl < zmax):
                        pairs.append((lvl, f"{int(p*100)}% "))
                pairs.sort(key=lambda t: t[0])  # strictly increasing for contour()

                if pairs:
                    levels_sorted = [lv for lv, _ in pairs]
                    fmt = {lv: lab for lv, lab in pairs}
                    cs = ax.contour(Xs, Ys, Z_mix, levels=levels_sorted, colors='white', linewidths=2)
                    ax.clabel(cs, fmt=fmt, inline=True, fontsize=12)
                else:
                    print(f"[warn] {name}: mixture is too flat for probability contours.")

                # (Optional) still show component centers
                for k in range(1, n_comp+1):
                    cx = result.params[f'g{k}_centerx'].value
                    cy = result.params[f'g{k}_centery'].value
                    ax.scatter([cx], [cy], c='red', s=90, marker='+')


                ax.set_title(f"Hitmap with {'Two' if n_comp==2 else 'One'} Gaussian Fit and σ-contours", fontsize=20)
                ax.tick_params(axis='both', labelsize=14)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=17))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=17))
                ax.set_xlabel('X-Axis (mm)', fontsize=20); ax.set_ylabel('Y-Axis (mm)', fontsize=20)
                fig.savefig(os.path.join(out_dir, f"{name}_with_contours.png"), dpi=200, bbox_inches="tight")
                plt.close(fig)

                # 8) CSV: one row per component
                csv_path = os.path.join(out_dir, f"{name}_fit.csv")
                with open(csv_path, 'w', newline='') as fh:
                    wcsv = csv.writer(fh, delimiter=';')
                    wcsv.writerow([
                        'component',
                        'amplitude','amplitude_error',
                        'centerx','centerx_error',
                        'centery','centery_error',
                        'sigma_x','sigmax_error',
                        'sigma_y','sigmay_error',
                        'fwhmx','fwhmx_error',
                        'fwhmy','fwhmy_error',
                        'bg_c'
                    ])
                    
                    for k in range(1, n_comp+1):
                        pre = f'g{k}_'
                        wcsv.writerow([
                            k,
                            _fmt_eu(result.params[pre+'amplitude'].value),
                            _fmt_eu(result.params[pre+'amplitude'].stderr),
                            _fmt_eu(result.params[pre+'centerx'].value),
                            _fmt_eu(result.params[pre+'centerx'].stderr),
                            _fmt_eu(result.params[pre+'centery'].value),
                            _fmt_eu(result.params[pre+'centery'].stderr),
                            _fmt_eu(result.params[pre+'sigmax'].value),
                            _fmt_eu(result.params[pre+'sigmax'].stderr),
                            _fmt_eu(result.params[pre+'sigmay'].value),
                            _fmt_eu(result.params[pre+'sigmay'].stderr),
                            _fmt_eu(result.params[pre+'fwhmx'].value),
                            _fmt_eu(result.params[pre+'fwhmx'].stderr),
                            _fmt_eu(result.params[pre+'fwhmy'].value),
                            _fmt_eu(result.params[pre+'fwhmy'].stderr),
                        ])

                print(f"→ {name}: fitted {n_comp} Gaussian component(s); results → {csv_path}")

        # In all modes, draw the plain subtracted 2D hist map (xyglob AND others)
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(Hd.T, origin='lower', extent=[-12.8,12.8,-12.8,12.8],  cmap='plasma')
        fig.colorbar(im, ax=ax, label='Counts')
        ax.set_aspect('auto')
        ax.set_title(f"{'Global Centroid Hitmap' if name=='xyglob' else name + '_sub' }")
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        fig.savefig(os.path.join(out_dir, f"{name}_sub.png"))
        plt.close(fig)

        # --- pixel maps subtraction ---
        if name == "pixmapglob":
            rng = [[-0.5, 7.5], [-0.5, 7.5]]
            w_s = spec.get("pixmapglob_w")
            w_b = bkg.get("pixmapglob_w")
            x_b, y_b = bkg["pixmapglob_x"], bkg["pixmapglob_y"]
        else:
            rng = [[-12.8, 12.8], [-12.8, 12.8]]
            w_s = w_b = None
            x_b, y_b = bkg[name+"_x"], bkg[name+"_y"]

        # build histograms with the RIGHT bins/ranges (use `bins` from hist2d_defs!)
        Hs, xe, ye = np.histogram2d(x_s, y_s, bins=bins, range=rng, weights=w_s)
        Hb, _, _   = np.histogram2d(x_b, y_b, bins=bins, range=rng, weights=w_b)
        Hd = Hs - Hb * scale
        Hd[Hd < 0] = 0
        Hd_plot = np.rot90(Hd, 90)
        # plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if name == "pixmapglob":
            im = ax.imshow(Hd.T, origin='lower', extent=[-0.5, 7.5, -0.5, 7.5], cmap='plasma', aspect='equal')
            ax.set_xlabel('Pixel X (index)'); ax.set_ylabel('Pixel Y (index)')
        else:
            im = ax.imshow(Hd.T, origin='lower', extent=[-12.8, 12.8, -12.8, 12.8], cmap='plasma')
            ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        fig.colorbar(im, ax=ax, label='Counts')
        ax.set_title(f"{'Global Centroid Hitmap' if name=='xyglob' else name + '_sub'}")
        fig.savefig(os.path.join(out_dir, f"{name}_sub.png"))
        plt.close(fig)
        # diff = spec[name] - bkg[name] * scale
        # diff[diff<0] = 0
        # fig, ax = plt.subplots()
        # im = ax.imshow(diff, origin='lower', extent=[0,8,0,8], cmap='plasma')
        # fig.colorbar(im, ax=ax, label='Counts')
        # ax.set_title(f"{name}_sub")
        # ax.set_xlabel('Channel ID X'); ax.set_ylabel('Channel ID Y')
        # fig.savefig(os.path.join(out_dir, f"{name}_sub.png"))
        # plt.close(fig)



if __name__ == "__main__":
    import sys

    # --- Interactive menu (1–4) ---
    print("Select analysis mode:")
    print(" 1 = Only histograms (no fitting)")
    print(" 2 = Fit only total_charge histogram")
    print(" 3 = Fit total_charge, xglob, and yglob; histograms only for 2D plots")
    print(" 4 = Full analysis (all fits and histograms)")
    try:
        choice = int(input("Enter your choice [1-4]: ").strip())
    except Exception:
        print("Invalid input, exiting.")
        sys.exit(1)
    if choice not in (1, 2, 3, 4):
        print("Invalid choice, exiting.")
        sys.exit(1)

    # parse CLI args
    p = argparse.ArgumentParser(
        description="Spectrum vs. Background subtraction & peak fitting"
    )
    p.add_argument("--bins1d-pos", type=int, default=None,
                   help="Number of bins for 1D position plots (xglob,yglob,xclus,yclus)")
    p.add_argument("--bins2d", type=int, default=None,
                   help="Number of bins per axis for 2D plots (xyglob,xyclus,xyglobloose,xyclusloose)")
    p.add_argument("--pixbins", type=int, default=None,
                   help="Number of bins per axis for the pixmapglob 2D histogram")
    p.add_argument("spectrum_file", help="ROOT file with spectrum")
    p.add_argument("spec_time",    type=float,
                   help="Live time (s) of spectrum run")
    p.add_argument("background_file", help="ROOT file with background")
    p.add_argument("bkg_time",       type=float,
                   help="Live time (s) of background run")
    args = p.parse_args()
        # Defaults if not provided on CLI; ask briefly so user can control binning
    bins1d_pos = args.bins1d_pos if args.bins1d_pos is not None else int(input("Bins for 1D positions [default 12]: ") or 12)
    bins2d     = args.bins2d     if args.bins2d     is not None else int(input("Bins per axis for 2D maps [default 12]: ") or 12)
    pixbins    = args.pixbins    if args.pixbins    is not None else int(input("Bins per axis for pixmapglob [default 8]: ") or 8)

    process_spectrum_minus_background(
        args.spectrum_file, args.spec_time,
        args.background_file, args.bkg_time,
        fit_mode=choice,
        bins1d_pos=bins1d_pos,
        bins2d=bins2d,
        pixbins=pixbins
    )
    
