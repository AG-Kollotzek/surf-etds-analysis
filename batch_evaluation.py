import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from DataConverter import ETDQAProcessor
import plotly.graph_objects as go

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
           'CSV': '163820'},
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
    """Binned JSON-Daten und berechnet korrekten Mittelwert/StdDev über mehrere Files."""

    # WICHTIG: bin_ms in Sekunden umrechnen! (200ms -> 0.2s)
    bin_sec = bin_ms / 1000.0

    aligned_frames = []

    for df in data_frames:
        # Erstelle eine Kopie, um Warnungen (SettingWithCopyWarning) zu vermeiden
        df_align = df.copy()

        # Da jedes File unterschiedlich lang vor dem ersten 5mm Peak herumdümpelt,
        # normalisieren wir die Zeitachse JEDES Files auf 0.0 beim ersten Frame.
        # Da DataConverter.py bereits alle Files auf "exakt 5s vor dem ersten Peak"
        # gecroppt hat, ist t=0 jetzt für alle Files der exakt gleiche relative Zeitpunkt!
        t_start = df_align['Time_Sec'].iloc[0]
        df_align['Time_Relative'] = df_align['Time_Sec'] - t_start

        aligned_frames.append(df_align)

    # Jetzt können wir sie gefahrlos übereinander legen
    combined = pd.concat(aligned_frames, ignore_index=True)

    # Berechnung des Bins in Sekunden (z.B. 0.2)
    # Runden ist hier sicherer als Modulo (//) bei Float-Zahlen
    combined['Time_Bin'] = np.round(combined['Time_Relative'] / bin_sec) * bin_sec

    # Gruppieren und Mitteln
    grouped = combined.groupby('Time_Bin')
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index().fillna(0)

    return mean_df, std_df


def calculate_rmsd(signal_exactrac, signal_brailab):
    """
    Berechnet die Root Mean Square Deviation zwischen zwei Signalen.
    Ignoriert NaNs, falls das Binning kleine Lücken gelassen hat.
    """
    # Filtere NaNs raus, damit die Mathe nicht crasht
    valid_idx = ~np.isnan(signal_exactrac) & ~np.isnan(signal_brailab)
    diff = signal_exactrac[valid_idx] - signal_brailab[valid_idx]

    # RMSD Formel: Wurzel aus dem Mittelwert der quadrierten Abweichungen
    rmsd = np.sqrt(np.mean(diff ** 2))
    return rmsd


def create_plot(mean_df, std_df, csv_df, title, is_translation=True, lost_times=None):
    """Erzeugt eine interaktive Plotly-Figure und annotiert den RMSD."""
    fig = go.Figure()
    rmsd_texts = []

    if is_translation:
        dof_map = {'lateral': 'True_Lateral', 'longitudinal': 'True_Longitudinal', 'vertical': 'True_Vertical'}
        # Format: (Linienfarbe, Schlauchfarbe ExacTrac)
        colors = {'lateral': ('red', 'rgba(255,0,0,0.3)'),
                  'longitudinal': ('green', 'rgba(0,128,0,0.3)'),
                  'vertical': ('blue', 'rgba(0,0,255,0.3)')}
        ylabel = 'Shift (mm)'
        unit = "mm"
    else:
        dof_map = {'pitch': 'True_Pitch', 'roll': 'True_Roll', 'yaw': 'True_Yaw'}
        colors = {'pitch': ('red', 'rgba(255,0,0,0.3)'),
                  'roll': ('green', 'rgba(0,128,0,0.3)'),
                  'yaw': ('blue', 'rgba(0,0,255,0.3)')}
        ylabel = 'Rotation (°)'
        unit = "°"

    # --- DER TIME-FIX ---
    csv_time = csv_df['Time_Sec'].values - csv_df['Time_Sec'].iloc[0]

    for etd_col, surf_col in dof_map.items():
        line_col, fill_etd = colors[etd_col]

        # 1. ExacTrac Daten vorbereiten
        x_etd = mean_df['Time_Bin'].values
        y_etd = mean_df[etd_col].values
        std_etd = std_df[etd_col].values

        # ExacTrac Schlauch
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_etd, x_etd[::-1]]),
            y=np.concatenate([y_etd + std_etd, (y_etd - std_etd)[::-1]]),
            fill='toself', fillcolor=fill_etd,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip', showlegend=False, name=f'ETD StdDev {etd_col}'
        ))

        # ExacTrac Mean Linie
        fig.add_trace(go.Scatter(
            x=x_etd, y=y_etd,
            mode='lines', line=dict(color=line_col, width=2),
            name=f'ExacTrac {etd_col}'
        ))

        # 2. Phantom Daten extrahieren (ohne Uncertainties-Fehler!)
        if f'{surf_col}_nominal' in csv_df.columns:
            csv_nom = csv_df[f'{surf_col}_nominal'].values
        else:
            csv_nom = np.array([getattr(v, 'n', v) for v in csv_df[surf_col]])

        # Phantom Linie
        fig.add_trace(go.Scatter(
            x=csv_time, y=csv_nom,
            mode='lines', line=dict(color=line_col, width=2, dash='dash'),
            name=f'Phantom {etd_col}'
        ))

        # 3. Interpolieren und RMSD berechnen
        # Da csv_time und x_etd unterschiedliche Zeitraster haben, interpolieren
        # wir die Phantom-Daten an die Zeitstempel der ExacTrac-Bins.
        interpolated_csv_nom = np.interp(x_etd, csv_time, csv_nom)

        rmsd_val = calculate_rmsd(y_etd, interpolated_csv_nom)
        rmsd_texts.append(f"{etd_col.capitalize()}: {rmsd_val:.3f} {unit}")

    # Zeichne die Tracking Lost Bereiche ein
    if lost_times:
        lost_times = sorted(list(set(lost_times)))
        intervals = []

        if lost_times:
            start_t = lost_times[0]
            prev_t = lost_times[0]
            # Verbinde nah beieinander liegende Lost-Frames (z.B. < 0.5s) zu einem Block
            for t in lost_times[1:]:
                if t - prev_t > 0.5:
                    intervals.append((start_t, prev_t))
                    start_t = t
                prev_t = t
            intervals.append((start_t, prev_t))

            # Füge für jedes identifizierte Intervall ein Rechteck in den Plot ein
            for (t0, t1) in intervals:
                # Mindestbreite für das Auge sichern, falls es nur ein einzelner Frame war
                t1 = max(t1, t0 + 0.1)
                fig.add_vrect(
                    x0=t0, x1=t1,
                    fillcolor="gray", opacity=0.3,
                    layer="below", line_width=0,
                    annotation_text="Tracking Lost", annotation_position="top left",
                    annotation_font_color="gray"
                )

    # Layout und Annotation
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white'
    )

    annotation_text = "<b>RMSD (ExacTrac vs. Surf Phantom):</b><br>" + "<br>".join(rmsd_texts)

    fig.add_annotation(
        x=0.98, y=0.02,  # Position: Fast ganz rechts (0.98) und fast ganz unten (0.02)
        xref="paper", yref="paper",  # Bezieht sich auf das gesamte Plot-Fenster (0 bis 1)
        text=annotation_text,
        showarrow=False,
        xanchor="right",  # Der rechte Rand der Box klebt an x=0.98
        yanchor="bottom",  # Der untere Rand der Box klebt an y=0.02
        font=dict(size=12, color="black"),
        align="left",  # Text innerhalb der Box bleibt linksbündig
        bgcolor="rgba(255, 255, 255, 0.7)",  # Etwas transparenter, damit man durchschimmernde Kurven noch sieht
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
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
        all_lost_times = []
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
                continue

            # Nimm jeweils den ersten Treffer (rglob ignoriert die Unterordner-Struktur davor)
            csv_path = str(csv_files[0])
            json_path = str(json_files[0])

            try:
                proc = ETDQAProcessor(terminal_version='legacy')
                proc.load_csv(csv_path)
                proc.load_json(json_path)
                proc.apply_kinematics(couch_angle=0.0)
                proc.align_signals()
                proc.apply_baseline_and_crop()

                all_json_dfs.append(proc.df_json)

                if hasattr(proc, 'lost_times_aligned'):
                    all_lost_times.extend(proc.lost_times_aligned)

                if reference_csv_df is None:
                    reference_csv_df = proc.df_csv

            except Exception as e:
                print(f"   [X] FEHLER bei ID {m_id}: {e}")
                group_failed = True
                continue

        if not all_json_dfs:
            print(f"   ---> Gruppe {gruppe_name} abgebrochen (Keine validen Daten).")
            continue

        # Mittelung und interaktives Plotting
        mean_df, std_df = bin_and_average(all_json_dfs)
        fig_trans = create_plot(mean_df, std_df, reference_csv_df, f"{gruppe_name} - Translation (Pads: {pad_status})",
                                True, all_lost_times)
        fig_rot = create_plot(mean_df, std_df, reference_csv_df, f"{gruppe_name} - Rotation (Pads: {pad_status})",
                              False, all_lost_times)

        fig_trans.show()
        fig_rot.show()

        cmd = input(
            f"\nBilder für {gruppe_name} speichern? ('save' zum Speichern, 'exit' zum Abbruch, 'Enter' zum Überspringen): ").strip().lower()

        if cmd == 'save':
            out_dir = results_base_dir / gruppe_name / f"Group_Pads_{pad_status}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Plotly nutzt write_image statt savefig
            try:
                fig_trans.write_image(out_dir / f"{gruppe_name}_Translation.png", scale=2)
                fig_rot.write_image(out_dir / f"{gruppe_name}_Rotation.png", scale=2)
                print(f"   [✓] Gespeichert in {out_dir}")
            except ValueError as e:
                print(f"   [!] Fehler beim Speichern (fehlt das 'kaleido' package?): {e}")
        elif cmd == 'exit':
            print("   [!] Verworfen. Beende Batch-Lauf.")
            break
        else:
            print("   [!] Übersprungen.")




if __name__ == "__main__":
    main()