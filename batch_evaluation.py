import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from DataConverter import ETDQAProcessor
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc

# --- 1. KONFIGURATION & MESSDATEN-STRUKTUR ---

# Automatisch aus dem Protokoll extrahierte Zuordnung
MEASUREMENT_10032026 = {
    '1': {'Gruppe': 'All axes', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '160617',
          'CSV': '160448', 'title': 'H, V, R'},
    '2': {'Gruppe': 'All axes', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161117',
          'CSV': '160950', 'title': 'H, V, R'},
    '3': {'Gruppe': 'All axes', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161416',
          'CSV': '161302', 'title': 'H, V, R'},
    '4': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161724',
          'CSV': '161538', 'title': 'H'},
    '5': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162054',
          'CSV': '161859', 'title': 'H'},
    '6': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162326',
          'CSV': '162130', 'title': 'H'},
    '7': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162647',
          'CSV': '162451', 'title': 'V'},
    '8': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162922',
          'CSV': '162728', 'title': 'V'},
    '9': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163157',
          'CSV': '162951', 'title': 'V'},
    '10': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163508',
           'CSV': '163325', 'title': 'R'},
    '11': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163758',
           'CSV': '163613', 'title': 'R'},
    '12': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '164007',
           'CSV': '163820', 'title': 'R'},
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
    '22': {'Gruppe': 'Vertical_Slide', 'Heatingpads': 'OFF', 'ETD': '182630', 'CSV': '182433'},
    '24': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190312',
           'CSV': '190114', 'title': 'H'},
    '25': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190609',
           'CSV': '190405', 'title': 'H'},
    '26': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190831',
           'CSV': '190639', 'title': 'H'},
    '27': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191212',
           'CSV': '191006', 'title': 'V'},
    '28': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191421',
           'CSV': '191235', 'title': 'V'},
    '29': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191651',
           'CSV': '191507', 'title': 'V'},
    '30': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192007',
           'CSV': '191806', 'title': 'R'},
    '31': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192224',
           'CSV': '192031', 'title': 'R'},
    '32': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192441',
           'CSV': '192250', 'title': 'R'},
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


import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_evaluation_results(
        t_kin, trans_et, trans_ihd, rot_et, rot_ihd,
        t_rmse, rmse3d, rmse_temp,
        plot_engine='matplotlib',
        save_path=None
):
    """
    Generates a synchronized 3-fold plot for evaluating one measurement group.

    Parameters:
    - t_kin: The synchronized (stretched/cut) time axis array for kinematics.
    - trans_et, trans_ihd: Dictionaries or DataFrames with keys 'X', 'Y', 'Z'
    - rot_et, rot_ihd: Dictionaries or DataFrames with keys 'pitch', 'yaw', 'roll'
    - t_rmse: The synchronized time axis array for the RMSE data. Must have identical limits to t_kin.
    - rmse3d, rmse_temp: Arrays containing the RMSE calculations.
    - plot_engine: 'matplotlib' (for publication) or 'plotly' (for exploration).
    """

    if plot_engine == 'matplotlib':
        # --- Formal Publication Styling ---
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 12,
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.2,
            'xtick.color': 'black',
            'ytick.color': 'black',
            'legend.frameon': True,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'legend.framealpha': 1.0,
            'legend.fontsize': 10
        })

        # sharex=True perfectly unifies the time axis across all three panels
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, dpi=300)
        line_w = 1.0  # Thinner lines to make dashed segments easily distinguishable

        # Consistent coloring across translational and rotational pairs
        colors = {'X_pitch': '#d62728', 'Y_yaw': '#1f77b4', 'Z_roll': '#2ca02c'}

        # --- Top Plot: Translations ---
        axes[0].plot(t_kin, trans_et['X'], label='ET X (lat)', color=colors['X_pitch'], linestyle='-', linewidth=line_w)
        axes[0].plot(t_kin, trans_ihd['X'], label='IHD X', color=colors['X_pitch'], linestyle='--', linewidth=line_w)

        axes[0].plot(t_kin, trans_et['Y'], label='ET Y (long)', color=colors['Y_yaw'], linestyle='-', linewidth=line_w)
        axes[0].plot(t_kin, trans_ihd['Y'], label='IHD Y', color=colors['Y_yaw'], linestyle='--', linewidth=line_w)

        axes[0].plot(t_kin, trans_et['Z'], label='ET Z (vert)', color=colors['Z_roll'], linestyle='-', linewidth=line_w)
        axes[0].plot(t_kin, trans_ihd['Z'], label='IHD Z', color=colors['Z_roll'], linestyle='--', linewidth=line_w)

        axes[0].set_ylabel('Translation [mm]', color='black')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # --- Middle Plot: Rotations ---
        axes[1].plot(t_kin, rot_et['pitch'], label='ET pitch', color=colors['X_pitch'], linestyle='-', linewidth=line_w)
        axes[1].plot(t_kin, rot_ihd['pitch'], label='IHD pitch', color=colors['X_pitch'], linestyle='--',
                     linewidth=line_w)

        axes[1].plot(t_kin, rot_et['yaw'], label='ET yaw', color=colors['Y_yaw'], linestyle='-', linewidth=line_w)
        axes[1].plot(t_kin, rot_ihd['yaw'], label='IHD yaw', color=colors['Y_yaw'], linestyle='--', linewidth=line_w)

        axes[1].plot(t_kin, rot_et['roll'], label='ET roll', color=colors['Z_roll'], linestyle='-', linewidth=line_w)
        axes[1].plot(t_kin, rot_ihd['roll'], label='IHD roll', color=colors['Z_roll'], linestyle='--', linewidth=line_w)

        axes[1].set_ylabel('Rotation [°]', color='black')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # --- Bottom Plot: RMSE ---
        axes[2].plot(t_rmse, rmse3d, label='RMSE3D', color='#9467bd', linestyle='-', linewidth=line_w)
        axes[2].plot(t_rmse, rmse_temp, label='RMSE_temp', color='#8c564b', linestyle='-', linewidth=line_w)

        axes[2].set_ylabel('RMSE', color='black')
        axes[2].set_xlabel('Time [s]', color='black')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')  # High res output
        plt.show()

    elif plot_engine == 'plotly':
        # --- Interactive Exploration Styling ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        colors = {'X_pitch': 'red', 'Y_yaw': 'blue', 'Z_roll': 'green'}
        line_w = 1.5

        # 1. Translation
        fig.add_trace(go.Scatter(x=t_kin, y=trans_et['X'], name='ET X (lat.)', mode='lines',
                                 line=dict(color=colors['X_pitch'], dash='solid', width=line_w)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=trans_ihd['X'], name='IHD X', mode='lines',
                                 line=dict(color=colors['X_pitch'], dash='dash', width=line_w)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=trans_et['Y'], name='ET Y (long.)', mode='lines',
                                 line=dict(color=colors['Y_yaw'], dash='solid', width=line_w)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=trans_ihd['Y'], name='IHD Y', mode='lines',
                                 line=dict(color=colors['Y_yaw'], dash='dash', width=line_w)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=trans_et['Z'], name='ET Z (vert.)', mode='lines',
                                 line=dict(color=colors['Z_roll'], dash='solid', width=line_w)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=trans_ihd['Z'], name='IHD Z', mode='lines',
                                 line=dict(color=colors['Z_roll'], dash='dash', width=line_w)), row=1, col=1)

        # 2. Rotation
        fig.add_trace(go.Scatter(x=t_kin, y=rot_et['pitch'], name='ET pitch', mode='lines',
                                 line=dict(color=colors['X_pitch'], dash='solid', width=line_w)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=rot_ihd['pitch'], name='IHD pitch', mode='lines',
                                 line=dict(color=colors['X_pitch'], dash='dash', width=line_w)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=rot_et['yaw'], name='ET yaw', mode='lines',
                                 line=dict(color=colors['Y_yaw'], dash='solid', width=line_w)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=rot_ihd['yaw'], name='IHD yaw', mode='lines',
                                 line=dict(color=colors['Y_yaw'], dash='dash', width=line_w)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=rot_et['roll'], name='ET roll', mode='lines',
                                 line=dict(color=colors['Z_roll'], dash='solid', width=line_w)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_kin, y=rot_ihd['roll'], name='IHD roll', mode='lines',
                                 line=dict(color=colors['Z_roll'], dash='dash', width=line_w)), row=2, col=1)

        # 3. RMSE
        fig.add_trace(go.Scatter(x=t_rmse, y=rmse3d, name='RMSE 3D', mode='lines',
                                 line=dict(color='#9467bd', dash='solid', width=line_w)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_rmse, y=rmse_temp, name='RMSE Temperature', mode='lines',
                                 line=dict(color='#8c564b', dash='solid', width=line_w)), row=3, col=1)

        # Format Plotly axes heavily to simulate publication readiness
        fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', gridcolor='lightgrey')

        fig.update_layout(
            font=dict(family="Arial", size=12, color="black"),
            plot_bgcolor='white',
            legend=dict(bgcolor="white", bordercolor="black", borderwidth=1),
            width=800, height=1000
        )
        fig.show()

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
                    layer="below", line_width=0
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


def plot_evaluation_results_interactive(
        t_kin, trans_et, trans_ihd, rot_et, rot_ihd,
        std_trans, std_rot,
        t_rmse, rmse3d, rmse_temp, save_path=None, group_name="Unbekannte Gruppe"
):
    """
    Creates an interactive 3-fold Matplotlib plot.
    Returns:
        str: 'saved' if the user pressed Enter, or 'exit' if the user typed 'exit'.
    """
    # --- Okabe-Ito Color Palette ---
    C_X_PITCH = '#D55E00'  # Vermillion
    C_Y_YAW = '#56B4E9'  # Sky Blue
    C_Z_ROLL = '#009E73'  # Bluish Green
    C_RMSE3D = '#CC79A7'  # Orange
    C_RMSETMP = '#E69F00'  # Reddish Purple

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
        'font.size': 12, 'axes.edgecolor': 'black', 'axes.linewidth': 1.2,
        'legend.frameon': True, 'legend.edgecolor': 'black'
    })

    # Auto-detect time regions based on Phantom movement
    movement_mag = np.abs(trans_ihd['X']) + np.abs(trans_ihd['Y']) + np.abs(trans_ihd['Z'])
    t_peak = t_kin[np.argmax(movement_mag)]

    window_size = max(1, int(len(movement_mag) / 20))
    min_var = float('inf')
    t_flat_idx = 0
    for i in range(0, len(movement_mag) - window_size, window_size):
        var = np.var(movement_mag[i:i + window_size])
        if var < min_var:
            min_var = var
            t_flat_idx = i + window_size // 2
    t_flat = t_kin[t_flat_idx]

    window_sec = 4.0

    # --- NEW: Mapping für Box-Positionen (x0, y0, width, height) ---
    POS_MAP = {
        'upper-left': [0.05, 0.60, 0.25, 0.35], 'top-left': [0.05, 0.60, 0.25, 0.35],
        'upper-right': [0.60, 0.60, 0.25, 0.35], 'top-right': [0.60, 0.60, 0.25, 0.35],
        'lower-left': [0.05, 0.05, 0.25, 0.35], 'bottom-left': [0.05, 0.05, 0.25, 0.35],
        'lower-right': [0.70, 0.05, 0.25, 0.35], 'bottom-right': [0.70, 0.05, 0.25, 0.35]
    }

    # State dictionary for the two zoom boxes
    zooms = {
        'peak': {
            'ax_idx': 0,
            'xlim': [max(0, t_peak - window_sec / 2), t_peak + window_sec / 2],
            'ylim': None,  # None means auto-scale
            'pos': POS_MAP['upper-left']
        },
        'flat': {
            'ax_idx': 0,
            'xlim': [max(0, t_flat - window_sec / 2), t_flat + window_sec / 2],
            'ylim': None,
            'pos': POS_MAP['upper-right']
        }
    }

    fig = None

    while True:
        if fig is not None:
            plt.close(fig)
            gc.collect()

        # Einmaliges Erstellen von Figure und Axes
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, dpi=100)

        # Haupttitel direkt hier setzen
        fig.suptitle(f"ExacTrac surface outputs vs. IHD (with moving axes: {group_name})",
                     fontsize=16, fontweight='bold')

        # Layout anpassen, damit der Titel nicht überlappt
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.98])

        line_w = 1.2

        # --- AXES 0: Translation ---
        # Phantom (Dashed)
        axes[0].plot(t_kin, trans_ihd['X'], label='IHD X', color=C_X_PITCH, linestyle='--', linewidth=line_w)
        axes[0].plot(t_kin, trans_ihd['Y'], label='IHD Y', color=C_Y_YAW, linestyle='--', linewidth=line_w)
        axes[0].plot(t_kin, trans_ihd['Z'], label='IHD Z', color=C_Z_ROLL, linestyle='--', linewidth=line_w)
        # ETD (Solid)
        axes[0].plot(t_kin, trans_et['X'], label='ET X (lat.)', color=C_X_PITCH, linestyle='-', linewidth=line_w)
        axes[0].plot(t_kin, trans_et['Y'], label='ET Y (long.)', color=C_Y_YAW, linestyle='-', linewidth=line_w)
        axes[0].plot(t_kin, trans_et['Z'], label='ET Z (vert.)', color=C_Z_ROLL, linestyle='-', linewidth=line_w)
        axes[0].set_ylabel('Translation [mm]')
        axes[0].legend(loc='upper right', ncol=1, fontsize=9)
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # --- NEU: Standardabweichungs-Bänder (Schläuche) ---
        axes[0].fill_between(t_kin, trans_et['X'] - std_trans['X'], trans_et['X'] + std_trans['X'],
                             color=C_X_PITCH, alpha=0.2, linewidth=0)
        axes[0].fill_between(t_kin, trans_et['Y'] - std_trans['Y'], trans_et['Y'] + std_trans['Y'],
                             color=C_Y_YAW, alpha=0.2, linewidth=0)
        axes[0].fill_between(t_kin, trans_et['Z'] - std_trans['Z'], trans_et['Z'] + std_trans['Z'],
                             color=C_Z_ROLL, alpha=0.2, linewidth=0)

        # --- AXES 1: Rotation ---
        # Phantom (Dashed)
        axes[1].plot(t_kin, rot_ihd['pitch'], label='IHD pitch', color=C_X_PITCH, linestyle='--', linewidth=line_w)
        axes[1].plot(t_kin, rot_ihd['yaw'], label='IHD yaw', color=C_Y_YAW, linestyle='--', linewidth=line_w)
        axes[1].plot(t_kin, rot_ihd['roll'], label='IHD roll', color=C_Z_ROLL, linestyle='--', linewidth=line_w)
        # ETD (Solid)
        axes[1].plot(t_kin, rot_et['pitch'], label='ET pitch', color=C_X_PITCH, linestyle='-', linewidth=line_w)
        axes[1].plot(t_kin, rot_et['yaw'], label='ET yaw', color=C_Y_YAW, linestyle='-', linewidth=line_w)
        axes[1].plot(t_kin, rot_et['roll'], label='ET roll', color=C_Z_ROLL, linestyle='-', linewidth=line_w)
        axes[1].set_ylabel('Rotation [°]')
        axes[1].legend(loc='upper right', ncol=1, fontsize=9)
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # --- NEU: Standardabweichungs-Bänder für Rotation ---
        axes[1].fill_between(t_kin, rot_et['pitch'] - std_rot['pitch'], rot_et['pitch'] + std_rot['pitch'],
                             color=C_X_PITCH, alpha=0.2, linewidth=0)
        axes[1].fill_between(t_kin, rot_et['yaw'] - std_rot['yaw'], rot_et['yaw'] + std_rot['yaw'],
                             color=C_Y_YAW, alpha=0.2, linewidth=0)
        axes[1].fill_between(t_kin, rot_et['roll'] - std_rot['roll'], rot_et['roll'] + std_rot['roll'],
                             color=C_Z_ROLL, alpha=0.2, linewidth=0)

        # --- AXES 2: RMSE ---
        axes[2].plot(t_rmse, rmse3d, label='RMSE 3D', color=C_RMSE3D, linestyle='-', linewidth=line_w)
        axes[2].plot(t_rmse, rmse_temp, label='RMSE Temp', color=C_RMSETMP, linestyle='-', linewidth=line_w)
        axes[2].set_ylabel('RMSE')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, linestyle=':', alpha=0.6)

        # --- Draw Zoom Boxes ---
        for z_name, z_data in zooms.items():
            ax_idx = z_data['ax_idx']
            ax_main = axes[ax_idx]

            axins = ax_main.inset_axes(z_data['pos'])

            if ax_idx == 0:
                axins.plot(t_kin, trans_ihd['X'], color=C_X_PITCH, linestyle='--')
                axins.plot(t_kin, trans_ihd['Y'], color=C_Y_YAW, linestyle='--')
                axins.plot(t_kin, trans_ihd['Z'], color=C_Z_ROLL, linestyle='--')
                axins.plot(t_kin, trans_et['X'], color=C_X_PITCH)
                axins.plot(t_kin, trans_et['Y'], color=C_Y_YAW)
                axins.plot(t_kin, trans_et['Z'], color=C_Z_ROLL)
                # NEU: Auch in der Zoom-Box die Schläuche zeichnen
                axins.fill_between(t_kin, trans_et['X'] - std_trans['X'], trans_et['X'] + std_trans['X'],
                                   color=C_X_PITCH, alpha=0.2)
                axins.fill_between(t_kin, trans_et['Y'] - std_trans['Y'], trans_et['Y'] + std_trans['Y'], color=C_Y_YAW,
                                   alpha=0.2)
                axins.fill_between(t_kin, trans_et['Z'] - std_trans['Z'], trans_et['Z'] + std_trans['Z'],
                                   color=C_Z_ROLL, alpha=0.2)
            elif ax_idx == 1:
                axins.plot(t_kin, rot_ihd['pitch'], color=C_X_PITCH, linestyle='--')
                axins.plot(t_kin, rot_ihd['yaw'], color=C_Y_YAW, linestyle='--')
                axins.plot(t_kin, rot_ihd['roll'], color=C_Z_ROLL, linestyle='--')
                axins.plot(t_kin, rot_et['pitch'], color=C_X_PITCH)
                axins.plot(t_kin, rot_et['yaw'], color=C_Y_YAW)
                axins.plot(t_kin, rot_et['roll'], color=C_Z_ROLL)
                # NEU: Auch in der Zoom-Box die Schläuche zeichnen
                axins.fill_between(t_kin, rot_et['X'] - std_rot['X'], rot_et['X'] + std_rot['X'],
                                   color=C_X_PITCH, alpha=0.2)
                axins.fill_between(t_kin, rot_et['Y'] - std_rot['Y'], rot_et['Y'] + std_rot['Y'], color=C_Y_YAW,
                                   alpha=0.2)
                axins.fill_between(t_kin, rot_et['Z'] - std_rot['Z'], rot_et['Z'] + std_rot['Z'],
                                   color=C_Z_ROLL, alpha=0.2)

            # Set X limits
            xlims = z_data['xlim']
            axins.set_xlim(xlims)

            # Apply manual or auto Y limits
            if z_data['ylim'] is not None:
                axins.set_ylim(z_data['ylim'])
            else:
                mask = (t_kin >= xlims[0]) & (t_kin <= xlims[1])
                if mask.any():
                    if ax_idx == 0:
                        y_vals = np.concatenate([trans_ihd['X'][mask], trans_ihd['Y'][mask], trans_ihd['Z'][mask],
                                                 trans_et['X'][mask], trans_et['Y'][mask], trans_et['Z'][mask]])
                    else:
                        y_vals = np.concatenate([rot_ihd['pitch'][mask], rot_ihd['yaw'][mask], rot_ihd['roll'][mask],
                                                 rot_et['pitch'][mask], rot_et['yaw'][mask], rot_et['roll'][mask]])
                    ymin, ymax = y_vals.min(), y_vals.max()
                    margin = max(0.1, (ymax - ymin) * 0.15)
                    axins.set_ylim(ymin - margin, ymax + margin)

            axins.set_xticklabels([])
            ax_main.indicate_inset_zoom(axins, edgecolor="black")

        plt.show(block=False)
        plt.pause(0.1)

        # --- Interactive Terminal Loop ---
        print("\n--- Plot Editor ---")
        print(" [Enter]    Save and continue")
        print(" [exit]     Abort batch evaluation")
        print(" [box off]  Entfernt alle Zoom-Boxen")
        print(" Edit Box format: [box] [axis] [tmin] [tmax] [ymin]* [ymax]* [position]*")
        print("          *ymin, ymax und position sind optional.")
        print("          Verfügbare Positionen: upper-left, upper-right, lower-left, lower-right")
        print(" Beispiele: 'peak 1 15.0 20.0'                   (Auto Y-Limits)")
        print("            'flat 0 10.0 15.0 -1 1'              (Manuelle Y-Limits)")
        print("            'peak 1 15.0 20.0 lower-right'       (Auto Limits + unten rechts)")
        print("            'flat 0 10.0 15.0 -1 1 upper-left'   (Manuelle Limits + oben links)")

        cmd = input("Command: ").strip().lower()

        if cmd == "exit":
            plt.close(fig)
            return 'exit'

        elif cmd == "":
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf', metadata={'Creator': 'MyEvaluationTool'})
                print(f"   [✓] Saved successfully to {save_path}")
            plt.close(fig)
            return 'saved'

        elif cmd == "box off":
            zooms.clear()
            print("   [i] Alle Zoom-Boxen wurden ausgeblendet.")

        else:
            try:
                parts = cmd.split()
                if len(parts) >= 4:
                    box_name, ax_idx = parts[0], int(parts[1])
                    tmin, tmax = float(parts[2]), float(parts[3])
                    ymin, ymax = None, None
                    new_pos = None

                    # Parse optionale Argumente am Ende des Strings
                    rem_parts = parts[4:]

                    # Prüfen, ob das letzte Wort ein Positions-Key ist
                    if rem_parts and rem_parts[-1] in POS_MAP:
                        new_pos = POS_MAP[rem_parts.pop()]

                    # Sind danach noch genau zwei Elemente übrig, sind es die y-Limits
                    if len(rem_parts) == 2:
                        ymin, ymax = float(rem_parts[0]), float(rem_parts[1])
                    elif len(rem_parts) != 0:
                        print(
                            "   [!] Ignoriere fehlerhafte Y-Limit Parameter. (Gib entweder beide Y-Werte an oder keinen)")

                    # Falls der User die Boxen zuvor mit "box off" gelöscht hat,
                    # müssen wir sie wieder im Dictionary initialisieren:
                    if box_name in ['peak', 'flat']:
                        if box_name not in zooms:
                            default_pos = POS_MAP['upper-left'] if box_name == 'peak' else POS_MAP['upper-right']
                            zooms[box_name] = {'pos': default_pos}

                        # Neue Werte zuweisen
                        zooms[box_name]['ax_idx'] = ax_idx
                        zooms[box_name]['xlim'] = [tmin, tmax]
                        zooms[box_name]['ylim'] = [ymin, ymax] if ymin is not None else None

                        # Position updaten, sofern angegeben
                        if new_pos is not None:
                            zooms[box_name]['pos'] = new_pos

                        print(f"Updating {box_name} box...")
                    else:
                        print("Invalid box name ('peak' or 'flat') or axis index (0 or 1).")
                else:
                    print("Invalid format. Too few arguments.")
            except Exception as e:
                print(f"Error parsing input: {e}. Please use the correct format.")
# --- 3. HAUPTAUSWERTUNG ---

def main():
    print("=== KONFIGURATION BATCH-EVALUIERUNG ===")
    print("Beispiele: 'all', 'all off', 'vertikal off', 'longitudinal 32 crop', 'other' für eigenen Dateipfad ")
    eval_input = input("Welchen Auswertungsmodus wählen?: ").strip().lower().split()

    if not eval_input:
        return

    # --- FEAT 1: Prüfen, ob "crop" gewünscht ist ---
    use_crop = False
    if "crop" in eval_input:
        use_crop = True
        eval_input.remove("crop")

    # --- FEAT 2: Custom Modus ("other") ---
    if eval_input[0] == "other":
        print("\n=== CUSTOM DATEIAUSWAHL (OTHER MODUS) ===")
        # Tkinter initialisieren, Fenster verstecken und on-top zwingen
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        print("-> Wähle die .csv Datei...")
        csv_path = filedialog.askopenfilename(title="SURF CSV auswählen", filetypes=[("CSV Files", "*.csv")])
        if not csv_path:
            print("Abbruch.")
            return

        print("-> Wähle die .json Datei...")
        json_path = filedialog.askopenfilename(title="ETD JSON auswählen", filetypes=[("JSON Files", "*.json")])
        if not json_path:
            print("Abbruch.")
            return

        root.destroy()  # Tkinter beenden

        print(f"\nLade:\n CSV: {Path(csv_path).name}\n JSON: {Path(json_path).name}")

        csv_sync_window = None
        if use_crop:
            crop_input = input("-> Crop-Fenster für CSV Sync (z.B. '10-50' in Sek, Leer=Überspringen): ").strip()
            if crop_input:
                try:
                    parts = crop_input.replace(',', '.').split('-')
                    t_min = float(parts[0])
                    t_max = float(parts[1]) if len(parts) > 1 else 99999.0
                    csv_sync_window = (t_min, t_max)
                except ValueError:
                    print("   [!] Ungültige Eingabe, verwende normales Alignment.")

        try:
            proc = ETDQAProcessor(terminal_version='legacy')
            proc.load_csv(csv_path)
            proc.load_json(json_path)
            proc.apply_kinematics(couch_angle=0.0)
            proc.align_signals(csv_sync_window=csv_sync_window)
            proc.apply_baseline_and_crop()

            # Auch ein einzelnes File kann an bin_and_average übergeben werden!
            mean_df, std_df = bin_and_average([proc.df_json])

            # NEU: Zeitachse der grauen Boxen an den 0.0-Startpunkt anpassen!
            t_start_json = proc.df_json['Time_Sec'].iloc[0]
            lost = [t - t_start_json for t in getattr(proc, 'lost_times_aligned', [])]

            fig_trans = create_plot(mean_df, std_df, proc.df_csv, f"Custom - Translation", True, lost)

            fig_rot = create_plot(mean_df, std_df, proc.df_csv, f"Custom - Rotation", False, lost)

            fig_trans.show()
            fig_rot.show()
        except Exception as e:
            print(f"Fehler bei der Custom-Auswertung: {e}")
            import traceback
            traceback.print_exc()

        return  # Bei "other" brechen wir hiernach ab, keine Batch-Schleife.

    # --- STANDARD BATCH MODUS ---
    target_group = eval_input[0]
    target_pads = eval_input[1].upper() if len(eval_input) >= 2 else "ALL"

    data_dir = Path('path/to/SURF/20260310_Messung_4/20260310_messung4')
    results_base_dir = Path('path/to/SURF/Paper_Ergebnisse')
    results_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarte Verarbeitung für Gruppe: '{target_group.upper()}', Heatingpads: '{target_pads}'")

    groups = {}
    for m_id, meta in MEASUREMENT_10032026.items():
        gruppe_lower = meta['Gruppe'].lower()
        pad_status = meta['Heatingpads']
        if target_group != "all" and gruppe_lower != target_group: continue
        if target_pads != "ALL" and pad_status != target_pads: continue
        key = (meta['Gruppe'], pad_status)
        if key not in groups: groups[key] = []
        groups[key].append(m_id)

    if not groups:
        print("Keine Messungen gefunden, die auf diesen Filter zutreffen!")
        return

    for (gruppe_name, pad_status), ids in groups.items():
        print(f"\n[{gruppe_name.upper()} | Pads: {pad_status}] ---> Verarbeite IDs: {ids}")
        all_json_dfs = []
        all_lost_times = []
        reference_csv_df = None

        for m_id in ids:
            meta = MEASUREMENT_10032026[m_id]
            etd_code, csv_code = meta['ETD'], meta['CSV']

            csv_files = list(data_dir.rglob(f"*{csv_code}.csv"))
            formatted_etd = f"{etd_code[:2]}-{etd_code[2:4]}-{etd_code[4:]}"
            json_files = list(data_dir.rglob(f"*{formatted_etd}.json"))

            if not csv_files or not json_files:
                print(f"   [!] FEHLT: Daten für ID {m_id}. Überspringe.")
                continue

            csv_path = str(csv_files[0])
            json_path = str(json_files[0])

            # NEU: CROP ABFRAGE PRO ID IM BATCH
            csv_sync_window = None
            if use_crop:
                crop_input = input(
                    f"   -> Crop-Fenster für ID {m_id} (z.B. '10' oder '10-50', Leer=Überspringen): ").strip()
                if crop_input:
                    try:
                        parts = crop_input.replace(',', '.').split('-')
                        t_min = float(parts[0])
                        t_max = float(parts[1]) if len(parts) > 1 else 99999.0
                        csv_sync_window = (t_min, t_max)
                    except ValueError:
                        print("      [!] Ungültige Eingabe, verwende normales Alignment.")

            try:
                proc = ETDQAProcessor(terminal_version='legacy')
                proc.load_csv(csv_path)
                proc.load_json(json_path)
                proc.apply_kinematics(couch_angle=0.0)
                proc.align_signals(csv_sync_window=csv_sync_window)
                proc.apply_baseline_and_crop()

                all_json_dfs.append(proc.df_json)
                if hasattr(proc, 'lost_times_aligned'):
                    # NEU: Auch hier müssen die Boxen auf 0.0 genullt werden!
                    t_start_json = proc.df_json['Time_Sec'].iloc[0]
                    shifted_lost = [t - t_start_json for t in proc.lost_times_aligned]
                    all_lost_times.extend(shifted_lost)
                if reference_csv_df is None:
                    reference_csv_df = proc.df_csv

            except Exception as e:
                print(f"   [X] FEHLER bei ID {m_id}: {e}")
                continue

        if not all_json_dfs:
            print(f"   ---> Gruppe {gruppe_name} abgebrochen (Keine validen Daten).")
            continue

        mean_df, std_df = bin_and_average(all_json_dfs)
        print(std_df.columns)
        t_kin = mean_df['Time_Bin'].values

        # Helper to interpolate CSV data to the binned JSON timeline
        def get_interp(col):
            csv_time = reference_csv_df['Time_Sec'].values - reference_csv_df['Time_Sec'].iloc[0]
            # Handle possible uncertainties (ufloat) vs standard floats
            if f'{col}_nominal' in reference_csv_df.columns:
                arr = reference_csv_df[f'{col}_nominal'].values
            else:
                arr = np.array([getattr(v, 'n', v) for v in reference_csv_df[col]])
            return np.interp(t_kin, csv_time, arr)

        # Dictionary structures for clean passing to plot function
        trans_et = {'X': mean_df['lateral'].values, 'Y': mean_df['longitudinal'].values, 'Z': mean_df['vertical'].values}
        trans_ihd = {'X': get_interp('True_Lateral'), 'Y': get_interp('True_Longitudinal'),
                     'Z': get_interp('True_Vertical')}
        rot_et = {'pitch': mean_df['pitch'].values, 'yaw': mean_df['yaw'].values, 'roll': mean_df['roll'].values}
        rot_ihd = {'pitch': get_interp('True_Pitch'), 'yaw': get_interp('True_Yaw'), 'roll': get_interp('True_Roll')}

        # RMSE Extraction (Using the columns we confirmed in previous steps)
        t_rmse = t_kin
        rmse3d_vals = mean_df['rmse3d'].values if 'rmse3d' in mean_df.columns else np.zeros_like(t_kin)
        rmse_temp_vals = mean_df['rmse_temp'].values if 'rmse_temp' in mean_df.columns else np.zeros_like(t_kin)

        print(f"\nGeneriere Plot für {gruppe_name} (Pads: {pad_status})...")

        # Determine save directory
        out_dir = results_base_dir / gruppe_name / f"Group_Pads_{pad_status}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{gruppe_name}_Combined_Evaluation.png"

        # Korrekt: Verwende std_df und die Original-Spaltennamen
        std_trans = {
            'X': std_df['lateral'].values,
            'Y': std_df['longitudinal'].values,
            'Z': std_df['vertical'].values
        }

        std_rot = {
            'pitch': std_df['pitch'].values,
            'yaw': std_df['yaw'].values,
            'roll': std_df['roll'].values
        }

        # Capture the result of the interactive session
        session_result = plot_evaluation_results_interactive(
            t_kin=t_kin,
            trans_et=trans_et,
            trans_ihd=trans_ihd,
            rot_et=rot_et,
            rot_ihd=rot_ihd,
            std_trans=std_trans,
            std_rot=std_rot,
            t_rmse=t_rmse,
            rmse3d=rmse3d_vals,
            rmse_temp=rmse_temp_vals,
            save_path=save_path,
            group_name= gruppe_name
        )

        if session_result == 'exit':
            print("\nAbbruch durch Benutzer ('exit').")
            break  # Breaks out of the main batch loop completely




if __name__ == "__main__":
    main()