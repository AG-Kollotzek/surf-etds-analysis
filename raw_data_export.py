import os
import json
import pandas as pd
from pathlib import Path
from DataConverter import ETDQAProcessor
from batch_evaluation import MEASUREMENT_10032026

# ==========================================
# KONFIGURATION
# ==========================================
BASE_EXPORT_DIR = Path("paper_data/process_export")
DATA_DIR = Path("paper_data")  # Pfad zu den JSON/CSV Rohdaten anpassen
ALLOWED_GROUPS = ['All axes', 'Horizontal', 'Vertical', 'Rotation']

# Die relevanten 7 Dimensionen
ETD_COLS = ['Time_Sec', 'lateral', 'longitudinal', 'vertical', 'pitch', 'yaw', 'roll']
PHANTOM_COLS = ['Time_Sec', 'True_Lateral', 'True_Longitudinal', 'True_Vertical', 'True_Pitch', 'True_Yaw', 'True_Roll']


def save_7d_array(df, columns, save_path):
    """Extrahiert die 7 Dimensionen und speichert sie sauber ab."""
    if df is not None and not df.empty:
        df_clean = df[columns].copy()
        df_clean.to_csv(save_path, index=False, sep=';', decimal='.')


def export_raw_data():
    for meas_id, meta in MEASUREMENT_10032026.items():
        gruppe = meta.get('Gruppe')

        # 1. Nur die relevanten Gruppen verarbeiten
        if gruppe not in ALLOWED_GROUPS:
            continue

        # Namensanpassung für "Longitudinal" -> "Horizontal" (wie gewünscht)
        if gruppe == 'Longitudinal':
            gruppe = 'Horizontal'

        # 2. Temperatur-Ordner bestimmen
        temp_folder = "32" if meta.get('Heatingpads') == '32' else "RT"

        # 3. Dateien lokalisieren (Dynamisch via rglob)
        etd_code, csv_code = str(meta['ETD']), str(meta['CSV'])

        # CSV suchen
        csv_files = list(DATA_DIR.rglob(f"*{csv_code}.csv"))

        # ETD formatieren (z.B. "123456" -> "12-34-56") und suchen
        if len(etd_code) >= 6:
            formatted_etd = f"{etd_code[:2]}-{etd_code[2:4]}-{etd_code[4:]}"
        else:
            formatted_etd = etd_code

        json_files = list(DATA_DIR.rglob(f"*{formatted_etd}.json"))

        # Fehlerbehandlung: Wenn Dateien fehlen, skippen
        if not csv_files or not json_files:
            print(f"   [!] FEHLT: Daten für ID {meas_id} ({gruppe}). Überspringe.")
            continue

        csv_path = str(csv_files[0])
        json_path = str(json_files[0])

        # 4. Zielordner erstellen
        out_dir = BASE_EXPORT_DIR / temp_folder / gruppe.replace(" ", "_") / f"meas_{str(meas_id).zfill(2)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Metadaten speichern
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)

        print(f"-> Verarbeite: {temp_folder} | {gruppe} | ID {meas_id} ...")

        try:
            # 5. Daten laden & Kinematik (STEP 1: BEFORE ALIGNMENT)
            proc = ETDQAProcessor(terminal_version='legacy')
            proc.load_csv(csv_path)
            proc.load_json(json_path)
            proc.apply_kinematics(couch_angle=0.0)

            # Zustand 1 abspeichern (Vor dem Alignen/Croppen)
            save_7d_array(proc.df_json, ETD_COLS, out_dir / "01_before_align_etd.csv")
            save_7d_array(proc.df_csv, PHANTOM_COLS, out_dir / "01_before_align_phantom.csv")

            # 6. Alignment & Cropping durchführen (STEP 2: AFTER ALIGNMENT)
            # Im automatischen Export wird das manuelle Crop-Fenster vorerst auf None belassen,
            # damit das Skript ohne Unterbrechung durchläuft.
            proc.align_signals(csv_sync_window=None)
            proc.apply_baseline_and_crop()

            # Zustand 2 abspeichern (Nach dem Alignen/Croppen)
            save_7d_array(proc.df_json, ETD_COLS, out_dir / "02_after_align_etd.csv")
            save_7d_array(proc.df_csv, PHANTOM_COLS, out_dir / "02_after_align_phantom.csv")

            print(f"   ✅ Erfolgreich exportiert.")

        except Exception as e:
            print(f"   [X] FEHLER bei ID {meas_id} während der Prozessierung: {e}")


if __name__ == "__main__":
    export_raw_data()