import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# KONFIGURATION (Konform zu raw_data_export.py)
# ==============================================================================
BASE_EXPORT_DIR = Path("paper_data/process_export")

# Zuordnung der Paare: Spaltenname Phantom (Baseline) vs. ETD Messung
DOF_PAIRS = {
    'lateral': ('True_Lateral', 'lateral'),
    'longitudinal': ('True_Longitudinal', 'longitudinal'),
    'vertical': ('True_Vertical', 'vertical'),
    'pitch': ('True_Pitch', 'pitch'),
    'yaw': ('True_Yaw', 'yaw'),
    'roll': ('True_Roll', 'roll')
}


def calculate_interpolated_rmse(meas_dir: Path):
    """
    Liest die 02_after_align_etd.csv und 02_after_align_phantom.csv aus meas_dir,
    interpoliert die ETD-Daten kontinuierlich auf die Phantom-Zeitstempel (Baseline)
    und berechnet den RMSE für alle 6 DoF. Das Ergebnis wird als 03_alldof_rmse.csv abgelegt.
    """
    etd_path = meas_dir / "02_after_align_etd.csv"
    phantom_path = meas_dir / "02_after_align_phantom.csv"

    if not etd_path.exists() or not phantom_path.exists():
        return None

    # Daten laden (Konvention: sep=';', decimal='.')
    df_etd = pd.read_csv(etd_path, sep=";")
    df_phantom = pd.read_csv(phantom_path, sep=";")

    if df_etd.empty or df_phantom.empty:
        print(f"[-] Warnung: Leere Dateien in {meas_dir}")
        return None

    # Zeitachsen extrahieren
    t_phantom = df_phantom['Time_Sec'].values
    t_etd = df_etd['Time_Sec'].values

    rmse_results = {}

    # Schleife über alle 6 Freiheitsgrade zur Interpolation und RMSE-Berechnung
    for dof_name, (phantom_col, etd_col) in DOF_PAIRS.items():
        phantom_vals = df_phantom[phantom_col].values
        etd_vals = df_etd[etd_col].values

        # Kontinuierliche lineare Interpolation von ETD auf die Phantom-Zeitstempel
        etd_vals_interpolated = np.interp(
            t_phantom,
            t_etd,
            etd_vals,
            left=etd_vals[0],
            right=etd_vals[-1]
        )

        # RMSE-Berechnung gegen die Phantom-Baseline
        diff = phantom_vals - etd_vals_interpolated
        rmse = np.sqrt(np.mean(diff ** 2))

        rmse_results[dof_name] = rmse

    # In DataFrame konvertieren
    df_rmse = pd.DataFrame([rmse_results])

    # Zielpfad definieren und abspeichern
    output_path = meas_dir / "03_alldof_rmse.csv"
    df_rmse.to_csv(output_path, index=False, sep=";", decimal=".")
    print(f"[+] RMSE berechnet & gespeichert: {output_path}")

    return rmse_results


def generate_group_summaries(all_measurements):
    """
    Gruppiert die gesammelten RMSE-Ergebnisse nach ihren übergeordneten Ordnern
    (z.B. RT/Horizontal) und berechnet pro DoF Mean und Std.
    """
    if not all_measurements:
        return

    # In DataFrame gießen für leichtere Gruppierung
    df_all = pd.DataFrame(all_measurements)

    # Gruppieren nach 'group_path' (das ist der Pfad bis zur Gruppe, z.B. paper_data/process_export/RT/Horizontal)
    for group_path, group_df in df_all.groupby('group_path'):
        summary_rows = []

        # Für jeden Freiheitsgrad Mean und Std berechnen
        for dof in DOF_PAIRS.keys():
            mean_val = group_df[dof].mean()
            std_val = group_df[dof].std()

            summary_rows.append({
                'DoF': dof,
                'RMSE_mean': mean_val,
                'RMSE_std': std_val
            })

        df_summary = pd.DataFrame(summary_rows)

        # Speicherpfad für die Gruppenzusammenfassung (04_group_rmse_summary.csv)
        output_path = Path(group_path) / "04_group_rmse_summary.csv"
        df_summary.to_csv(output_path, index=False, sep=";", decimal=".")
        print(f"[#] Gruppen-Zusammenfassung gespeichert: {output_path}")


def main():
    print(f"Starte RMSE-Berechnung im Verzeichnis: {BASE_EXPORT_DIR}...")
    if not BASE_EXPORT_DIR.exists():
        print(f"[-] Fehler: Verzeichnis {BASE_EXPORT_DIR} existiert nicht.")
        return

    count = 0
    all_measurements_data = []

    # Gehe rekursiv durch alle Unterordner (Gruppe/Pads/meas_XX)
    for root, dirs, files in os.walk(BASE_EXPORT_DIR):
        root_path = Path(root)
        if root_path.name.startswith("meas_"):
            rmse_dict = calculate_interpolated_rmse(root_path)
            if rmse_dict:
                # Ergänze den Pfad der übergeordneten Messgruppe (z.B. .../RT/Horizontal)
                rmse_dict['group_path'] = str(root_path.parent)
                all_measurements_data.append(rmse_dict)
                count += 1

    print(f"\n[Fertig] Einzel-RMSE für {count} Messordner erfolgreich berechnet.")

    # Schritt 2: Zusammenfassung pro Gruppe aggregieren und speichern
    print("Generiere Gruppen-Zusammenfassungen (Mean & Std)...")
    generate_group_summaries(all_measurements_data)
    print("[Komplett Fertig]")


if __name__ == "__main__":
    main()