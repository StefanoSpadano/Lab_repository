import argparse
import numpy as np
import awkward as ak
import uproot
import logging
import re
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

#Automatically fixes working directory by placing the the repository as the root 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
logging.info(f"Working directory automatically fixed at: {ROOT_DIR}")

def _parse_and_collect(ifile, gain="LG"):
    """
    Reads a single txt file and builds lists for the TTree.
    Returns: base_name_clean, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG
    """
    trig_ids = []
    trig_times = []
    all_chan_ids = []
    all_data_LG = []
    all_data_HG = []

    chan_ids_evt = []
    data_LG_evt = []
    data_HG_evt = []
    current_trig_id = None
    current_trig_time = None
    event_counter = 0

    with open(ifile, "r") as f:
        for line in f:
            if "Tstamp_us" in line or "//" in line:
                continue
            parts = line.split()
            if not parts:
                continue
            timecheck = parts[0]

            #continuation line (per-pixel)
            if "." not in timecheck:
                #push pixel data for the current event
                if gain == "LG":
                    channel = int(parts[1]); datum = int(parts[2])
                    data_LG_evt.append(datum)
                elif gain == "HG":
                    channel = int(parts[1]); datum = int(parts[4])
                    data_HG_evt.append(datum)
                else:
                    channel = int(parts[1])
                    datumLG = int(parts[2]); datumHG = int(parts[3])
                    data_LG_evt.append(datumLG); data_HG_evt.append(datumHG)
                chan_ids_evt.append(channel)
            else:
                #new event header line -> flush previous event if exists
                if event_counter > 0 and current_trig_id is not None:
                    trig_ids.append(current_trig_id)
                    trig_times.append(current_trig_time)
                    all_chan_ids.append(chan_ids_evt)
                    all_data_LG.append(data_LG_evt)
                    all_data_HG.append(data_HG_evt)

                #reset per-event buffers
                chan_ids_evt = []
                data_LG_evt = []
                data_HG_evt = []

                # BUG FIX 11/03/2026: only metadata from each header event is extracted, 
                #ignoring the remaining values as they are not real pixels. 
                current_trig_time = float(timecheck)
                current_trig_id = int(parts[1])
                event_counter += 1

    #write last event
    if event_counter > 0 and current_trig_id is not None:
        trig_ids.append(current_trig_id)
        trig_times.append(current_trig_time)
        all_chan_ids.append(chan_ids_evt)
        all_data_LG.append(data_LG_evt)
        all_data_HG.append(data_HG_evt)

    base_name = os.path.splitext(os.path.basename(ifile))[0]
    #clean name: remove '_list' and date-like patterns if present -> keep RunN
    clean_name = re.sub(r"(_?\d{4}[-_]\d{2}[-_]\d{2}|_?list)", "", base_name, flags=re.IGNORECASE)
    clean_name = clean_name.strip("_")
    if clean_name == "":
        clean_name = base_name

    return clean_name, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG

def _write_root(ofile, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG):
    """
    Writes the TTree ROOT. Each branch needs to be of the same length. 
    """
    n_events = len(trig_ids)
    #computes run_time_sec in seconds starting from the differences in the timestamps
    if n_events >= 2:
        run_time_sec = (float(trig_times[-1]) - float(trig_times[0])) * 1e-6
    else:
        run_time_sec = 0.0

    #run_time_sec is repeated for each event to prevent future shape errors
    run_time_array = np.full(n_events, run_time_sec, dtype=np.float64)

    arr = {
        "trigID": np.array(trig_ids, dtype=np.uint32),
        "trigTime": np.array(trig_times, dtype=np.float64),
        "channelID": ak.Array(all_chan_ids),
        "channelDataLG": ak.Array(all_data_LG),
        "channelDataHG": ak.Array(all_data_HG),
        "run_time_sec": run_time_array
    }

    #file root being written
    with uproot.recreate(ofile) as root_file:
        root_file["fersTree"] = arr

    return n_events, run_time_sec, trig_times[0] if len(trig_times) else None, trig_times[-1] if len(trig_times) else None

def convert_single_file(ifile, out_dir, gain="LG"):
    if not os.path.exists(ifile):
        raise FileNotFoundError(f"Input file not found: {ifile}")

    os.makedirs(out_dir, exist_ok=True)
    clean_name, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG = _parse_and_collect(ifile, gain=gain)

    #output name: RunN.root
    out_file = os.path.join(out_dir, f"{clean_name}.root")

    n_events = 0
    run_time_sec = 0.0
    first_ts = last_ts = None
    if len(trig_ids) == 0:
        logging.warning(f"No extracted events from {ifile}. Still builds the empty ROOT file to maintain cosistency.")
        #empty TTree for compatibility
        _ = _write_root(out_file, [], [], [], [], [])
    else:
        n_events, run_time_sec, first_ts, last_ts = _write_root(out_file, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG)

    logging.info(f"ROOT file successfully saved: {out_file}")
    logging.info(f"  events: {n_events}, run_time_sec: {run_time_sec:.6f} s (first={first_ts}, last={last_ts})")
    return out_file

def convert_folder_date(date_str=None, src_base="Data"):
    """
    If not date_str given, it searches for the last folder in Data\ resembling a date (YYYY or YYYY-MM-DD or YYYY_MM_DD).
    Converts all Run*_list.txt files inside that folder. 
    Output in Data/Data_converted/<date_str>/RunN.root
    """
    data_root = os.path.join(ROOT_DIR, "Data")
    if date_str is None:
        #finds folders in Data\ 
        cand = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        #orders for data or last updates.
        cand_sorted = sorted(cand, key=lambda x: os.path.getmtime(os.path.join(data_root, x)), reverse=True)
        if not cand_sorted:
            raise FileNotFoundError("No folder found in Data/")
        chosen = cand_sorted[0]
        date_str = chosen
        logging.info(f"No date given: picking the last folder created in Data/: {date_str}")
    else:
        if not os.path.isdir(os.path.join(data_root, date_str)):
            raise FileNotFoundError(f"Folder Data/{date_str} not found.")

    src_dir = os.path.join(data_root, date_str)
    target_dir = os.path.join(ROOT_DIR, "Data", "Data_converted", date_str)
    os.makedirs(target_dir, exist_ok=True)

    #finds the file Run*_list*.txt
    files = sorted([os.path.join(src_dir, f) for f in os.listdir(src_dir)
                    if re.match(r"(?i)^Run\d+.*list.*\.txt$", f)])
    if not files:
        logging.warning(f"No Run*_list.txt found in {src_dir}")
        return []

    out_files = []
    for f in files:
        try:
            outf = convert_single_file(f, target_dir)
            out_files.append(outf)
        except Exception as e:
            logging.error(f"Error converting {f}: {e}")

    return out_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FERS .txt -> ROOT (RunN.root) in Data/Data_converted/<date>/")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--input-file", "-i", help="File path .txt (es. Data/.../Run0_list.txt)")
    group.add_argument("--date", "-d", help="Folder name data in Data/ (es. 2025-12-02). If omitted picks the last created folder in Data\)")
    parser.add_argument("--gain", choices=["LG", "HG", "BOTH"], help="Gain (if not specified)")
    args = parser.parse_args()

    if args.input_file:
        #Converts a single file, saved in Data/Data_converted/<date_from_filepath> if possibile
        inpath = os.path.abspath(args.input_file)
        base_name = os.path.splitext(os.path.basename(inpath))[0]
        #if the path has a folder data it will be used as date_str
        parent = os.path.basename(os.path.dirname(inpath))
        date_match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", parent)
        if date_match:
            date_str = parent
        else:
            #fallback: saves directly in Data_converted root
            date_str = None

        if args.gain:
            gain = args.gain
        else:
            gm = re.search(r"(LG|HG|BOTH)", base_name, re.IGNORECASE)
            gain = gm.group(1).upper() if gm else "LG"

        if date_str:
            out_dir = os.path.join(ROOT_DIR, "Data", "Data_converted", date_str)
        else:
            out_dir = os.path.join(ROOT_DIR, "Data", "Data_converted")
        os.makedirs(out_dir, exist_ok=True)
        convert_single_file(inpath, out_dir, gain=gain)
    else:
        #converts the whole folder
        convert_folder_date(date_str=args.date)


