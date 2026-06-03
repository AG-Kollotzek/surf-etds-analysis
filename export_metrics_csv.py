import os
import pandas as pd
import numpy as np
from pathlib import Path
from uncertainties import ufloat
from uncertainties import unumpy as unp
from DataConverter import ETDQAProcessor

# Importiere die Konfiguration direkt aus deiner funktionierenden batch_evaluation
from batch_evaluation import MEASUREMENT_10032026


# ==========================================
# 1. METRIK-BERECHNUNG & FORMATIERUNG
# ==========================================
def calculate_max_dev_werror(etd_signal, phantom_nominal, total_error):
    """Berechnet die maximale Abweichung inklusive kombiniertem Fehler."""
    phantom_werror = unp.uarray(phantom_nominal, total_error)
    diff = unp.fabs(etd_signal - phantom_werror)
    max_idx = np.argmax(unp.nominal_values(diff))
    return diff[max_idx]


def calculate_rmsd_werror(etd_signal, phantom_nominal, total_error):
    """Berechnet den RMSD über den gesamten Verlauf inklusive kombiniertem Fehler."""
    phantom_werror = unp.uarray(phantom_nominal, total_error)
    mse = np.mean((etd_signal - phantom_werror) ** 2)
    return unp.sqrt(mse)


def format_uval(uval, decimals=3):
    """Konvertiert ein ufloat exakt in das Format: 'Wert (Unsicherheit)'"""
    # Falls Numpy das ufloat in ein 0D- oder 1-Element-Array eingepackt hat, packen wir es aus:
    if isinstance(uval, np.ndarray):
        uval = uval.item()

    return f"{uval.nominal_value:.{decimals}f} ({uval.std_dev:.{decimals}f})"


# ==========================================
# 2. HILFSFUNKTIONEN FÜR INTERPOLATION & MITTELUNG
# ==========================================
def get_interp_data(df_json, df_csv, t_common):
    """Interpoliert JSON und CSV (inkl. systematischer Fehler-Tubes) auf ein Zeitraster."""
    interp_results = {}

    # ETD (JSON) Signale
    for col in ['lateral', 'longitudinal', 'vertical', 'pitch', 'yaw', 'roll']:
        interp_results[f'et_{col}'] = np.interp(t_common, df_json['Time_Sec'], df_json[col])

    # Phantom (CSV) Nominal- und systematische Fehlerwerte (_std)
    csv_mapping = {
        'lateral': 'True_Lateral', 'longitudinal': 'True_Longitudinal', 'vertical': 'True_Vertical',
        'pitch': 'True_Pitch', 'roll': 'True_Roll', 'yaw': 'True_Yaw'
    }

    for short_name, csv_name in csv_mapping.items():
        interp_results[f'ihd_{short_name}_nom'] = np.interp(t_common, df_csv['Time_Sec'], df_csv[csv_name])
        interp_results[f'ihd_{short_name}_sys_err'] = np.interp(t_common, df_csv['Time_Sec'], df_csv[f"{csv_name}_std"])

    return interp_results


def bin_and_average_measurements(processed_runs):
    """Mittelt die Durchgänge und extrahiert den statistischen Fehler zueinander."""
    mean_curves = {}
    stat_errors = {}

    keys = [
        'et_lateral', 'et_longitudinal', 'et_vertical', 'et_pitch', 'et_yaw', 'et_roll',
        'ihd_lateral_nom', 'ihd_longitudinal_nom', 'ihd_vertical_nom', 'ihd_pitch_nom', 'ihd_roll_nom', 'ihd_yaw_nom',
        'ihd_lateral_sys_err', 'ihd_longitudinal_sys_err', 'ihd_vertical_sys_err', 'ihd_pitch_sys_err',
        'ihd_roll_sys_err', 'ihd_yaw_sys_err'
    ]

    for key in keys:
        matrix = np.array([run[key] for run in processed_runs])
        mean_curves[key] = np.mean(matrix, axis=0)

        # Statistischer Fehler (Standardabweichung der Läufe untereinander)
        if not key.endswith('_sys_err'):
            clean_key = key.replace('et_', '').replace('ihd_', '').replace('_nom', '')
            stat_errors[clean_key] = np.std(matrix, axis=0)

    return mean_curves, stat_errors


# ==========================================
# 3. KREUZTABELLENSTRUKTUR INITIALISIEREN
# ==========================================
dofs = ['X ( Lat.) / mm', 'Y (Long.) / mm', 'Z ( Vert.) / mm', 'Yaw  / °', 'Pitch / °', 'Roll / °']
temps = ['RT', '32']
groups = ['IHD: horizontal Axis', 'IHD: vertical Axis', 'IHD: rotational Axis']

columns = pd.MultiIndex.from_product([dofs, temps], names=['DOF', 'Temp of Heatingpads'])
df_rmsd = pd.DataFrame(index=groups, columns=columns)
df_maxdev = pd.DataFrame(index=groups, columns=columns)

group_mapping = {'Longitudinal': 'IHD: horizontal Axis', 'Vertical': 'IHD: vertical Axis',
                 'Rotation': 'IHD: rotational Axis'}
dof_mapping = {
    'lateral': 'X ( Lat.) / mm', 'longitudinal': 'Y (Long.) / mm', 'vertical': 'Z ( Vert.) / mm',
    'yaw': 'Yaw  / °', 'pitch': 'Pitch / °', 'roll': 'Roll / °'
}

# ==========================================
# 4. CONFIG & RUNTIME SETTINGS
# ==========================================
data_dir = Path('path/to/SURF/20260310_Messung_4/20260310_messung4') # Pfad zu deinen Datenordnern anpassen
use_crop = False  # Auf True setzen, falls du das Crop-Fenster interaktiv abfragen willst

# ==========================================
# 5. VERARBEITUNGSSCHLEIFE (JETZT DYNAMISCH)
# ==========================================
for gruppe_name in ['Longitudinal', 'Vertical', 'Rotation']:
    row_label = group_mapping[gruppe_name]

    for pad_status in ['OFF', '32']:
        temp_label = 'RT' if pad_status == 'OFF' else '32'

        matching_ids = [
            m_id for m_id, meta in MEASUREMENT_10032026.items()
            if meta['Gruppe'] == gruppe_name and meta['Heatingpads'] == pad_status
        ]

        if not matching_ids:
            continue

        print(f"\n[➔] Starte Batch für: {gruppe_name} (Pads: {pad_status})")

        # --- PHASE 1: Alle Läufe prozessieren und im Speicher halten ---
        processed_procs = []

        for m_id in matching_ids:
            meta = MEASUREMENT_10032026[m_id]
            etd_code, csv_code = meta['ETD'], meta['CSV']

            csv_files = list(data_dir.rglob(f"*{csv_code}.csv"))
            formatted_etd = f"{etd_code[:2]}-{etd_code[2:4]}-{etd_code[4:]}"
            json_files = list(data_dir.rglob(f"*{formatted_etd}.json"))

            if not csv_files or not json_files:
                print(f"   [!] FEHLT: Daten für ID {m_id}. Überspringe.")
                continue

            try:
                proc = ETDQAProcessor(terminal_version='legacy')
                proc.load_csv(str(csv_files[0]))
                proc.load_json(str(json_files[0]))
                proc.apply_kinematics(couch_angle=0.0)
                proc.align_signals(csv_sync_window=None)  # oder dein interaktives Fenster
                proc.apply_baseline_and_crop()

                # Prozessor-Instanz merken, statt sofort zu interpolieren
                processed_procs.append(proc)

            except Exception as e:
                print(f"   [X] FEHLER bei ID {m_id}: {e}")
                continue

        if not processed_procs:
            continue

        # --- PHASE 2: DYNAMISCHE TIMELINE FÜR DIESE GRUPPE BERECHNEN ---
        # Wir suchen die Schnittmenge (Overlapping Region) aller Läufe nach dem Croppen.
        # Das garantiert, dass wir für die RMSD-Berechnung bei jedem Zeitschritt echte Daten von JEDEM Lauf haben.
        t_start = max(p.df_json['Time_Sec'].min() for p in processed_procs)
        t_end = min(p.df_json['Time_Sec'].max() for p in processed_procs)

        # Feste Schrittweite von 0.1 Sekunden (entspricht deinen ~10 Hz Kamera-Frequenz)
        dt = 0.2
        num_steps = int((t_end - t_start) / dt) + 1
        t_common = np.linspace(t_start, t_end, num_steps)

        print(f"   -> Dynamische Timeline generiert: {t_start:.2f}s bis {t_end:.2f}s ({num_steps} Bins)")

        # Jetzt erst auf die maßgeschneiderte Timeline interpolieren
        runs_in_group = []
        for proc in processed_procs:
            interp_run = get_interp_data(proc.df_json, proc.df_csv, t_common)
            runs_in_group.append(interp_run)

        # --- AB HIER BLEIBT ALLES GLEICH (Mittelung & RMSD) ---
        mean_curves, stat_errors = bin_and_average_measurements(runs_in_group)

        for dof_key, col_label in dof_mapping.items():
            etd_sig = mean_curves[f'et_{dof_key}']
            phantom_nom = mean_curves[f'ihd_{dof_key}_nom']

            sigma_stat = stat_errors[dof_key]
            sigma_sys = mean_curves[f'ihd_{dof_key}_sys_err']
            total_error = np.sqrt(sigma_stat ** 2 + sigma_sys ** 2)

            rmsd_uval = calculate_rmsd_werror(etd_sig, phantom_nom, total_error)
            maxdev_uval = calculate_max_dev_werror(etd_sig, phantom_nom, total_error)

            df_rmsd.loc[row_label, (col_label, temp_label)] = format_uval(rmsd_uval)
            df_maxdev.loc[row_label, (col_label, temp_label)] = format_uval(maxdev_uval)

# ==========================================
# 6. EXPORT
# ==========================================
df_rmsd.to_csv('Paper_RMSD_Evaluation.csv')
df_maxdev.to_csv('Paper_MaxDev_Evaluation.csv')

print(
    "\n[✔] Fertig! Die Tabellen 'Paper_RMSD_Evaluation.csv' und 'Paper_MaxDev_Evaluation.csv' wurden erfolgreich erstellt.")