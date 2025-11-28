import os
import matplotlib.pyplot as plt

from analysis.io_manager import (
    get_project_root,
    get_latest_data_folder,
    list_run_files
)

from analysis.event_processing import collect_all_histograms
from analysis.plotting import (
    plot_histogram_with_fit,
    plot_heatmap,
    plot_heatmap_with_fit2D
)
from analysis.save_results import save_all_results


def main():

    print("\n=== STEP 1: Individuazione directory dati ===")

    root = get_project_root()
    data_base = os.path.join(root, "Data", "Data_converted")
    latest = get_latest_data_folder(data_base)
    runs = list_run_files(latest)

    print("Ultima cartella dati:", latest)
    print("Run trovate:", runs)

    if len(runs) == 0:
        print("❌ Nessuna run trovata.")
        return

    date_str = os.path.basename(latest)

    # ======================================================
    # PROCESSA TUTTE LE RUN PRESENTI NELLA CARTELLA
    # ======================================================
    for run_number, run_path in runs.items():

        print(f"\n\n=== PROCESSO RUN {run_number} ===")
        print(f"File: {run_path}")

        # 1) Processa gli eventi
        res = collect_all_histograms(run_path)

        # 2) Genera figure
        fig1, ax1 = plot_histogram_with_fit(
            *res["histograms"]["charge"],
            res["fits"]["charge"]
        )
        fig2, ax2 = plot_heatmap(res["maps"]["pixmapglob"])
        fig3, ax3 = plot_heatmap_with_fit2D(
            res["maps"]["pixmapclus"],
            res["fits"]["spot2d"]
        )

        figs = {
            "hist_charge": fig1,
            "map_global": fig2,
            "map_cluster_fit": fig3
        }

        # 3) Salvataggio completo
        save_all_results(res, figs, date_str, run_number)

        # chiudi figure per sicurezza
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    # ======================================================
    print("\n\n=== PIPELINE COMPLETATA PER TUTTE LE RUN ===")
    print("Risultati salvati in:")
    print(os.path.join(root, "Results", date_str))


if __name__ == "__main__":
    main()
