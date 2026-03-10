import os
import sys
import argparse
import subprocess
import json
import time
import numpy as np
from analysis.io_manager import get_data_converted_dir, get_project_root
from analysis.event_processing import collect_all_histograms
from analysis.save_results import save_all_results

# --- UTILITY PER LANCIARE SCRIPT ---
def run_module(module_name, args=[]):
    """Lancia un modulo python come sottoprocesso mantenendo l'ambiente."""
    print(f"\n🚀 Avvio modulo: {module_name}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    cmd = [sys.executable, "-m", module_name] + args
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"✅ {module_name} completato.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore in {module_name}: {e}")
        return False

# --- GESTIONE BACKGROUND DINAMICO ---
def generate_local_background(data_dir):
    """
    Simula il comportamento di create_bkg_ref.py ma on-the-fly.
    Cerca una cartella 'Background' dentro la cartella dati.
    """
    bkg_dir = os.path.join(data_dir, "Background")
    
    # Se la cartella non esiste, pazienza.
    if not os.path.exists(bkg_dir):
        print("ℹ️  Nessuna cartella 'Background' trovata locale. Procedo senza sottrazione.")
        return None

    # Cerca file .root
    files = [f for f in os.listdir(bkg_dir) if f.endswith(".root")]
    if not files:
        print("⚠️  Cartella 'Background' esiste ma è vuota. Procedo senza sottrazione.")
        return None

    bkg_file = files[0] # Prendiamo il primo file trovato
    bkg_path = os.path.join(bkg_dir, bkg_file)
    
    print(f"\n🧹 CREAZIONE RIFERIMENTO BACKGROUND LOCALE")
    print(f"    File: {bkg_file}")
    
    # --- PUNTO CRUCIALE: Usiamo event_processing con bkg_data=None ---
    raw_bkg_results = collect_all_histograms(bkg_path, bkg_data=None)
    
    if not raw_bkg_results:
        print("❌ Errore processamento Background. Sottrazione annullata.")
        return None

    # Verifica veloce durata (giusto per sicurezza)
    dur = raw_bkg_results.get("acquisition_time_sec", 0)
    print(f"    ✅ Background elaborato in memoria (Durata: {dur:.1f}s)")
    
    return raw_bkg_results

# --- PROCESSING ---
def process_run_data(run_path, date_str, bkg_data):
    run_name = os.path.splitext(os.path.basename(run_path))[0]
    
    # Sicurezza: se per caso c'è un file "background.root" nella cartella principale, ignoralo
    if "background" in run_name.lower(): 
        return

    print(f"   -> Processing: {run_name}...")
    
    # Qui passiamo bkg_data. Se non è None, event_processing chiamerà subtract_background_from_results
    results = collect_all_histograms(run_path, bkg_data)
    
    if not results:
        print(f"      ⚠️ Nessun risultato per {run_name}. Salto.")
        return
    
    save_all_results(results, date_str, run_name)

# --- DASHBOARD (Disattivata ma mantenuta nel codice per il futuro) ---
def show_dashboard_and_pick_run(date_str):
    root_dir = get_project_root()
    json_path = os.path.join(root_dir, "Results", date_str, "thesis_summary_data.json")
    
    if not os.path.exists(json_path):
        print("\n❌ File summary non trovato.")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    runs = sorted(data.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
    
    print("\n" + "="*60)
    print(f"📊 DASHBOARD RISULTATI (Crystal Ball - CLUSTER 5x5) - {date_str}")
    print(f"{'RUN':<10} | {'RES (%)':<10} | {'CHI2':<10} | {'MU (ADU)':<10}")
    print("-" * 60)
    
    valid_runs = []
    for r in runs:
        try:
            if "Cluster" in data[r]:
                res = data[r]["Cluster"]["res"]
                chi2 = data[r]["Cluster"]["chi2"]
                mu = data[r]["Cluster"]["mu"]
                print(f"{r:<10} | {res:<10.2f} | {chi2:<10.2f} | {mu:<10.0f}")
                valid_runs.append(r)
            else:
                print(f"{r:<10} | {'NO CLUSTER':<10} | {'-':<10} | {'-':<10}")
        except:
            print(f"{r:<10} | {'ERR':<10} | {'-':<10} | {'-':<10}")

    print("-" * 60)
    while True:
        choice = input("👉 Inserisci nome Run (es. Run9) o 'q' per uscire: ").strip()
        if choice.lower() == 'q': return None
        if choice in valid_runs: return choice
        print("⚠️ Run non valida. Riprova.")

def main():
    parser = argparse.ArgumentParser(description="Full Analysis Pipeline (ONLY PROCESSING)")
    parser.add_argument("date", type=str, help="Cartella data (es. 27_11_2025)")
    parser.add_argument("--skip-process", action="store_true", help="Salta la fase di elaborazione dati raw")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"🔬 ISOLPHARM DATA PROCESSOR - {args.date}")
    print(f"{'='*60}")

    # --- FASE 0: PROCESSING DATI RAW ---
    if not args.skip_process:
        print("\n--- FASE 0: ELABORAZIONE DATI RAW ---")
        data_dir = get_data_converted_dir(args.date)
        
        if os.path.exists(data_dir):
            # 1. Generazione Dinamica Background
            bkg_data = generate_local_background(data_dir)
            
            if bkg_data: 
                print("✅ Background Reference generato. Sarà sottratto a tutte le run.")
            
            # 2. Processing Runs
            files = [f for f in os.listdir(data_dir) if f.endswith(".root")]
            if files:
                print(f"📂 Trovati {len(files)} file .root da analizzare")
                for f in files:
                    process_run_data(os.path.join(data_dir, f), args.date, bkg_data)
            else:
                print("❌ Nessun file .root trovato nella cartella principale.")
        else:
            print(f"❌ Cartella dati non trovata: {data_dir}")
    else:
        print("\n⏭️  Salto Fase 0 (Processing)...")

    # =========================================================================
    # LE FASI SUCCESSIVE SONO DISATTIVATE PER CONSENTIRE IL FIT MANUALE
    # =========================================================================
    
    # --- FASE 1: FITTING ---
    # print("\n--- FASE 1: FITTING MODELLI ---")
    # run_module("analysis.fit_scipy_cb", [args.date])
    # run_module("analysis.fit_scipy_gmm", [args.date])

    # --- FASE 2: PLOTTING ---
    # print("\n--- FASE 2: GENERAZIONE GRAFICI ---")
    # run_module("analysis.summary_cb", [args.date])
    # run_module("analysis.summary_gmm", [args.date])
    # run_module("analysis.plot_physics_check", [args.date])

    # --- FASE 3: SYSTEMATICS ---
    # print("\n--- FASE 3: STUDIO SISTEMATICHE ---")
    # selected_run = show_dashboard_and_pick_run(args.date)
    # if selected_run:
    #     print(f"\n🌟 Avvio Systematics su: {selected_run}")
    #     run_module("analysis.systematics_cb", [args.date, selected_run])
    #     run_module("analysis.systematics_gmm", [args.date, selected_run])
    #     print("\n✅ Systematics completate.")
    # else:
    #     print("\n⏭️  Systematics saltate.")

    print(f"\n{'='*60}")
    print("✅ ELABORAZIONE DATI RAW COMPLETATA.")
    print("   Ora puoi lanciare i fit manualmente:")
    print(f"   👉 python -m analysis.fit_scipy_cb {args.date}")
    print(f"   👉 python -m analysis.fit_scipy_gmm {args.date}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()