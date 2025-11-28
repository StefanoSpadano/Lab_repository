#!/usr/bin/env python3
import argparse
import numpy as np
import awkward as ak
import uproot
import logging
import re
import os
from datetime import datetime

# Choose level and format of the log
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# --- Fix working directory automatically ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
print(f"[INFO] Working directory fissata automaticamente a: {ROOT_DIR}")


def convert_txt_to_root(ifilename, ofilename, gain="LG"):
    """
    Parse the text file and write out a ROOT file with a single TTree 'fersTree'.
    
    Parameters
    ----------
    ifilename : str
        Path to input ASCII file.
    ofilename : str
        Path to output .root file.
    gain : {'LG','HG','BOTH'}
        Which gain channels to record.
    """
    # Containers for all events
    trig_ids       = []
    trig_times     = []
    all_chan_ids   = []
    all_data_LG    = []
    all_data_HG    = []
    
    # Temporary per-event buffers
    chan_ids_evt   = []
    data_LG_evt    = []
    data_HG_evt    = []
    current_trig_id   = None
    current_trig_time = None
    
    event_counter = 0
    with open(ifilename, 'r') as f:
        for line in f:
            # skip headers and comments
            if "Tstamp_us" in line or "//" in line:
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            timecheck = parts[0]
            # continuation of current event?
            if "." not in timecheck:
                # push channel-by-channel data
                if gain == "LG":
                    channel = int(parts[1]); datum = int(parts[2])
                    data_LG_evt.append(datum)
                elif gain == "HG":
                    channel = int(parts[1]); datum = int(parts[4])
                    data_HG_evt.append(datum)
                else:  # BOTH
                    channel = int(parts[1])
                    datumLG = int(parts[2]); datumHG = int(parts[3])
                    data_LG_evt.append(datumLG)
                    data_HG_evt.append(datumHG)
                chan_ids_evt.append(channel)
            
            else:
                # new event: first, save previous (if any)
                if event_counter > 0:
                    trig_ids.append(current_trig_id)
                    trig_times.append(current_trig_time)
                    all_chan_ids.append(chan_ids_evt)
                    all_data_LG.append(data_LG_evt)
                    all_data_HG.append(data_HG_evt)
                
                # reset buffers
                chan_ids_evt   = []
                data_LG_evt    = []
                data_HG_evt    = []
                
                # parse new-event line
                current_trig_time = float(timecheck)
                if gain == "LG":
                    # time, trig_id, board, channel, datumLG
                    current_trig_id = int(parts[1])
                    channel = int(parts[3]); datum = int(parts[4])
                    data_LG_evt.append(datum)
                elif gain == "HG":
                    # time, trig_id, board, channel, S1, datumHG
                    current_trig_id = int(parts[1])
                    channel = int(parts[3]); datum = int(parts[5])
                    data_HG_evt.append(datum)
                else:  # BOTH
                    # time, trig_id, board, channel, datumLG, datumHG
                    current_trig_id = int(parts[1])
                    channel = int(parts[3])
                    datumLG = int(parts[4]); datumHG = int(parts[5])
                    data_LG_evt.append(datumLG)
                    data_HG_evt.append(datumHG)
                
                chan_ids_evt.append(channel)
                event_counter += 1
    
    # write the last event
    if event_counter > 0 and current_trig_id is not None:
        trig_ids.append(current_trig_id)
        trig_times.append(current_trig_time)
        all_chan_ids.append(chan_ids_evt)
        all_data_LG.append(data_LG_evt)
        all_data_HG.append(data_HG_evt)
    
    # prepare arrays / jagged arrays
    arr = {
        "trigID":     np.array(trig_ids,   dtype=np.uint32),
        "trigTime":   np.array(trig_times, dtype=np.float64),
        "channelID":      ak.Array(all_chan_ids),
        "channelDataLG":  ak.Array(all_data_LG),
        "channelDataHG":  ak.Array(all_data_HG),
    }
    
    # write to ROOT
    with uproot.recreate(ofilename) as root_file:
        root_file["fersTree"] = arr
        logging.info(f"File ROOT salvato con successo: {ofilename}")



if __name__ == "__main__":
    # setup logging 
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    parser = argparse.ArgumentParser(description="Convert FERS .txt to ROOT")
    parser.add_argument("input_file", help="Percorso al file .txt")
    parser.add_argument(
        "--gain", choices=["LG", "HG", "BOTH"],
        help="Gain da usare (opzionale, se non specificato tenta di dedurlo dal nome del file)"
    )
    args = parser.parse_args()

    # base name of the source file
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    # base directory to store converted files
    base_output_dir = os.path.join("Data", "Data_converted")

    # check if the file has a date already in its name (format AAAA_MM_GG or AAAA-MM-GG)
    date_match = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2})", base_name)

    if date_match:
        # if a date exists then create a folder to store files for that date
        date_str = date_match.group(1).replace("-", "_")  # uniform separators
        out_dir = os.path.join(base_output_dir, date_str)
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f"Trovata data nel nome del file: {date_str} → creerò la cartella {out_dir}")

        # cleaning before final name
        clean_name = re.sub(r"(_?\d{4}[-_]\d{2}[-_]\d{2}|_?list)", "", base_name, flags=re.IGNORECASE)
        output_file = os.path.join(out_dir, f"{clean_name}.root")

    else:
        # if no date then save it in Data_converted as it is
        out_dir = base_output_dir
        os.makedirs(out_dir, exist_ok=True)
        logging.info("Nessuna data trovata nel nome del file: salvataggio diretto in Data_converted/")

        output_file = os.path.join(out_dir, f"{base_name}_LG.root")

    # check if the name comes with a different type of gain from LG otherwise use LG
    if args.gain:
        gain = args.gain
    else:
        gain_match = re.search(r"(LG|HG|BOTH)", base_name, re.IGNORECASE)
        gain = gain_match.group(1).upper() if gain_match else "LG"
        logging.info(f"Gain dedotto automaticamente: {gain}")

    logging.info(f"Starting conversion: {args.input_file} → {output_file}")

    try:
        convert_txt_to_root(args.input_file, output_file, gain)
        logging.info(f"File ROOT creato con successo: {output_file}")
    except Exception as e:
        logging.error(f"Errore durante la conversione: {e}")



