import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from DataConverter import ETDQAProcessor

# --- 1. KONFIGURATION & MESSDATEN-STRUKTUR ---

# Automatisch aus dem Protokoll extrahierte Zuordnung
MEASUREMENT_10032026 = {
    '1': {'Gruppe': 'Standardmessung', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '160617',
          'CSV': '160448'},
    '2': {'Gruppe': 'Standardmessung', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161117',
          'CSV': '160950'},
    '3': {'Gruppe': 'Standardmessung', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161416',
          'CSV': '161302'},
    '4': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161724',
          'CSV': '161538'},
    '5': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162054',
          'CSV': '161859'},
    '6': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162326',
          'CSV': '162130'},
    '7': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162647',
          'CSV': '162451'},
    '8': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162922',
          'CSV': '162728'},
    '9': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163157',
          'CSV': '162951'},
    '10': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163508',
           'CSV': '163325'},
    '11': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163758',
           'CSV': '163613'},
    '12': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '164007',
           'CSV': '172148'},
    '14': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '173928',
           'CSV': '173645'},
    '15': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '174334',
           'CSV': '174056'},
    '16': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '174832',
           'CSV': '174558'},
    '17': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'Fitting', 'Heatingpads': 'OFF', 'ETD': '175259',
           'CSV': '175019'},
    '18': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'OnlyFrontSurface', 'Heatingpads': 'OFF', 'ETD': '175635',
           'CSV': '175356'},
    '19': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'Fiting', 'Heatingpads': 'OFF', 'ETD': '180108',
           'CSV': '175824'},
    '21': {'Gruppe': 'Vertical_Slide', 'Heatingpads': 'OFF', 'ETD': '182337', 'CSV': '182011'},
    '22': {'Gruppe': 'Vertical_Slide', 'Heatingpads': 'OFF', 'ETD': '194758', 'CSV': '193557'},
    '24': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190312',
           'CSV': '190114'},
    '25': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190609',
           'CSV': '190405'},
    '26': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190831',
           'CSV': '190639'},
    '27': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191212',
           'CSV': '191006'},
    '28': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191421',
           'CSV': '191235'},
    '29': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191651',
           'CSV': '191507'},
    '30': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192007',
           'CSV': '191806'},
    '31': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192224',
           'CSV': '192031'},
    '32': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192441',
           'CSV': '192250'},
    '33': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192853',
           'CSV': '192615'},
    '34': {'Gruppe': 'Vertical_Slide', 'Heatingpads': '32', 'ETD': '193528', 'CSV': '193140'},
    '35': {'Gruppe': 'Vertical_Slide', 'Heatingpads': '32', 'ETD': '193801', 'CSV': '193557'},
}


# --- 2. HILFSFUNKTIONEN ---

def check_tracking_lost(json_path):
    """Prüft das JSON auf das Signal 'trackingLost'."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    for frame in data.get("trackingResults", []):
        if frame.get("trackingLost") is True:
            return True
    return False


def bin_and_average(data_frames, bin_ms=200):
    """Binned JSON-Daten in 200ms Schritte und berechnet Mittelwert/StdDev."""
    combined = pd.concat(data_frames, ignore_index=True)
    combined['Time_Bin'] = (combined['timestamp'] // bin_ms) * bin_ms
    grouped = combined.groupby('Time_Bin')
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index().fillna(0)
    return mean_df, std_df


def create_plot(mean_df, std_df, csv_df, title, is_translation=True):
    """Erzeugt eine Figure für die Daten, zeigt sie aber noch nicht an."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_translation:
        dofs = ['lateral', 'longitudinal', 'vertical']
        colors = {'lateral': 'red', 'longitudinal': 'green', 'vertical': 'blue'}
        ylabel = 'Shift (mm)'
    else:
        dofs = ['pitch', 'roll', 'yaw']
        colors = {'pitch': 'red', 'roll': 'green', 'yaw': 'blue'}
        ylabel = 'Rotation (°)'

    for dof in dofs:
        color = colors[dof]
        # ExacTrac Daten
        ax.plot(mean_df['Time_Bin'], mean_df[dof], color=color, label=f'ExacTrac {dof}')
        ax.fill_between(mean_df['Time_Bin'], mean_df[dof] - std_df[dof], mean_df[dof] + std_df[dof], color=color,
                        alpha=0.3)

        # CSV Daten (mit oder ohne uncertainties package)
        csv_time = csv_df['timestamp']
        if f'{dof}_nominal' in csv_df.columns:
            csv_nom, csv_std = csv_df[f'{dof}_nominal'], csv_df[f'{dof}_std']
        else:
            csv_nom = np.array([getattr(v, 'n', v) for v in csv_df[dof]])
            csv_std = np.array([getattr(v, 's', 0) for v in csv_df[dof]])

        ax.plot(csv_time, csv_nom, color=color, linestyle='--', label=f'Phantom {dof}')
        ax.fill_between(csv_time, csv_nom - csv_std, csv_nom + csv_std, color=color, alpha=0.1, linestyle='--')

    ax.set_title(title)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    return fig


# --- 3. HAUPTAUSWERTUNG ---

def main():
    print("=== KONFIGURATION BATCH-EVALUIERUNG ===")
    print("Beispiele für Eingaben: 'all', 'all off', 'vertikal off', 'longitudinal 32'")
    eval_input = input("Welchen Auswertungsmodus wählen?: ").strip().lower().split()

    # Defaults
    target_group = "all"
    target_pads = "all"

    # Eingabe parsen (z.B. ['vertikal', 'off'] oder ['all'])
    if len(eval_input) == 1:
        target_group = eval_input[0]
    elif len(eval_input) >= 2:
        target_group = eval_input[0]
        target_pads = eval_input[1].upper()  # Pad Status immer Uppercase für den Match

    # Basis-Verzeichnisse anpassen!
    data_dir = Path('path/to/SURF/20260310_Messung_4/20260310_messung4')
    results_base_dir = Path('path/to/SURF/Paper_Ergebnisse')
    results_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarte Verarbeitung für Gruppe: '{target_group.upper()}', Heatingpads: '{target_pads}'")

    # Messungen filtern und gruppieren
    groups = {}
    for m_id, meta in MEASUREMENT_10032026.items():
        gruppe_lower = meta['Gruppe'].lower()
        pad_status = meta['Heatingpads']

        # Prüfe Gruppen-Filter
        if target_group != "all" and gruppe_lower != target_group:
            continue

        # Prüfe Pad-Filter
        if target_pads != "ALL" and pad_status != target_pads:
            continue

        key = (meta['Gruppe'], pad_status)
        if key not in groups:
            groups[key] = []
        groups[key].append(m_id)

    if not groups:
        print("Keine Messungen gefunden, die auf diesen Filter zutreffen!")
        return

    # Gefilterte Gruppen abarbeiten
    for (gruppe_name, pad_status), ids in groups.items():
        print(f"\n[{gruppe_name.upper()} | Pads: {pad_status}] ---> Verarbeite IDs: {ids}")
        all_json_dfs = []
        reference_csv_df = None
        group_failed = False

        for m_id in ids:
            meta = MEASUREMENT_10032026[m_id]
            etd_code = meta['ETD']  # z.B. "160617"
            csv_code = meta['CSV']  # z.B. "160448"

            # 1. CSV-Suche: Orientiert sich nur an den letzten 6 Ziffern vor .csv
            csv_files = list(data_dir.rglob(f"*{csv_code}.csv"))

            # 2. JSON-Suche: Wandelt "160617" in "16-06-17" um, damit es zum Dateinamen passt
            formatted_etd = f"{etd_code[:2]}-{etd_code[2:4]}-{etd_code[4:]}"
            json_files = list(data_dir.rglob(f"*{formatted_etd}.json"))

            if not csv_files or not json_files:
                print(f"   [!] FEHLT: Daten für ID {m_id}. Gruppe wird übersprungen.")
                group_failed = True
                break

            # Nimm jeweils den ersten Treffer (rglob ignoriert die Unterordner-Struktur davor)
            csv_path = str(csv_files[0])
            json_path = str(json_files[0])

            if check_tracking_lost(json_path):
                print(f"   [!] ABBRUCH: Tracking Lost in ID {m_id} gefunden!")
                group_failed = True
                break

            try:
                proc = ETDQAProcessor(terminal_version='legacy')
                proc.load_csv(csv_path)
                proc.load_json(json_path)
                proc.apply_kinematics(couch_angle=0.0)
                proc.align_signals()
                proc.apply_baseline_and_crop()

                all_json_dfs.append(proc.json_df)
                if reference_csv_df is None:
                    reference_csv_df = proc.csv_df

            except Exception as e:
                print(f"   [X] FEHLER bei ID {m_id}: {e}")
                group_failed = True
                break

        if group_failed or not all_json_dfs:
            print(f"   ---> Gruppe {gruppe_name} abgebrochen.")
            continue

        # Mittelung und interaktives Plotting
        mean_df, std_df = bin_and_average(all_json_dfs)
        fig_trans = create_plot(mean_df, std_df, reference_csv_df, f"{gruppe_name} - Translation (Pads: {pad_status})",
                                True)
        fig_rot = create_plot(mean_df, std_df, reference_csv_df, f"{gruppe_name} - Rotation (Pads: {pad_status})",
                              False)

        plt.show()  # Zeigt beide Fenster an und pausiert Skript

        cmd = input(
            f"\nBilder für {gruppe_name} speichern? ('save' zum Speichern, 'exit' zum Abbruch, 'Enter' zum Überspringen): ").strip().lower()
        if cmd == 'save':
            out_dir = results_base_dir / gruppe_name / f"Group_Pads_{pad_status}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_trans.savefig(out_dir / f"{gruppe_name}_Translation.png")
            fig_rot.savefig(out_dir / f"{gruppe_name}_Rotation.png")
            print(f"   [✓] Gespeichert in {out_dir}")
        elif cmd == 'exit':
            print("   [!] Verworfen. Beende Batch-Lauf.")
            plt.close('all')
            break
        else:
            print("   [!] Übersprungen.")

        plt.close('all')


if __name__ == "__main__":
    main()