import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from Kinematics import SurfKinematics
from uncertainties import unumpy as unp

class ETDQAProcessor:
    def __init__(self, terminal_version='legacy'):
        """
        terminal_version:
            'legacy' -> Invertiert Pos_H und Pos_R (für alte Datensätze)
            'v2'     -> Nutzt Rohdaten wie sie sind (für zukünftige korrigierte Version)
        """
        self.df_csv = None
        self.df_json = None
        self.time_offset = 0
        self.terminal_version = terminal_version

    def load_csv(self, file_path):
        """Lädt das SURF-Terminal Log und korrigiert ggf. Vorzeichenfehler"""
        self.df_csv = pd.read_csv(file_path, sep=';', decimal='.')

        # --- SMART SIGN CORRECTION ---
        if self.terminal_version == 'legacy':
            # Korrektur für Longitudinal (H) und Rotation (R)
            self.df_csv['Pos_H'] = self.df_csv['Pos_H'] * -1
            self.df_csv['Pos_R'] = self.df_csv['Pos_R'] * -1
            print(f"INFO: Legacy-Mode aktiv. Vorzeichen für Pos_H und Pos_R wurden invertiert.")

        return self.df_csv

    def load_json(self, file_path):
        """Lädt das ETD-Tracking Log und berechnet die 3D-Magnitude"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        results = []
        for entry in data['trackingResults']:
            if not entry.get('trackingLost', False):
                shifts = {k: float(v) for k, v in entry['shiftValues'].items()}
                shifts['timestamp_ms'] = entry['timestamp']
                results.append(shifts)

        self.df_json = pd.DataFrame(results)
        self.df_json['Time_Sec'] = (self.df_json['timestamp_ms'] - self.df_json['timestamp_ms'].iloc[0]) / 1000.0

        # Internes Alignment nutzt Magnitude (unabhängig vom Vorzeichen)
        lat0, long0, vert0 = self.df_json['lateral'].iloc[0], self.df_json['longitudinal'].iloc[0], \
        self.df_json['vertical'].iloc[0]
        self.df_json['Vector_Mag'] = np.sqrt(
            (self.df_json['lateral'] - lat0) ** 2 +
            (self.df_json['longitudinal'] - long0) ** 2 +
            (self.df_json['vertical'] - vert0) ** 2
        )
        return self.df_json

    def apply_kinematics(self, couch_angle=0.0):
        """Berechnet klinische Koordinaten und bereitet Fehler-Tubes vor (OHNE Baseline-Nullung)."""
        print("Berechne kinematische Transformation in klinische Koordinaten...")
        kin = SurfKinematics()

        # Berechnung (ohne Nullen)
        res = kin.calculate_task_space(self.df_csv['Pos_H'].values,
                                       self.df_csv['Pos_V'].values,
                                       self.df_csv['Pos_R'].values, couch_angle)

        for key, uarray in res.items():
            self.df_csv[key] = unp.nominal_values(uarray)
            self.df_csv[f"{key}_std"] = unp.std_devs(uarray)
            self.df_csv[f"{key}_upper"] = self.df_csv[key] + self.df_csv[f"{key}_std"]
            self.df_csv[f"{key}_lower"] = self.df_csv[key] - self.df_csv[f"{key}_std"]

    def align_signals(self, sync_axis='Pos_H', threshold=4.5):
        """Synchronisiert und speichert die exakten Zeiten der Peaks für das spätere Cropping."""

        def get_peak_info(times, values, thresh):
            v_smooth = pd.Series(values).rolling(window=5, center=True).median().fillna(0).values
            mask = np.abs(v_smooth) > thresh

            # Finde Kanten
            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            # Korrektur: Wir nehmen den Index des letzten 'True' Werts
            ends = np.where(edges == -1)[0] - 1

            if len(starts) < 2:
                return None, None, None, None

            # --- BIAS-FIX: FLOAT-PRÄZISION STATT INT-RUNDUNG ---
            # Wir nehmen den zeitlichen Mittelpunkt zwischen Start und Ende der Flanken
            t_first_mid = (times.iloc[starts[0]] + times.iloc[ends[0]]) / 2.0
            t_last_mid = (times.iloc[starts[-1]] + times.iloc[ends[-1]]) / 2.0

            t_first_start = times.iloc[starts[0]]
            t_last_end = times.iloc[ends[-1]]

            return t_first_mid, t_last_mid, t_first_start, t_last_end

        # Peak-Zeiten ermitteln
        csv_first, csv_last, csv_start, csv_end = get_peak_info(self.df_csv['Time_Sec'], self.df_csv[sync_axis],
                                                                threshold)
        json_first, json_last, _, _ = get_peak_info(self.df_json['Time_Sec'], self.df_json['Vector_Mag'], threshold)

        if csv_first is None or json_first is None:
            raise ValueError("Nicht genügend 5mm Peaks für Sync gefunden.")

        scale_factor = (csv_last - csv_first) / (json_last - json_first)
        self.df_json['Time_Sec'] = (self.df_json['Time_Sec'] - json_first) * scale_factor + csv_first
        self.time_offset = 0

        # WICHTIG: Start und Ende für die Baseline-Korrektur speichern
        self.sync_t_start = csv_start
        self.sync_t_end = csv_end

        print(f"Sync erfolgreich (Skalierung: {scale_factor:.6f}, Referenz-Peak: {csv_first:.2f}s)")
        return scale_factor

    def apply_baseline_and_crop(self):
        """Nullt die Achsen präzise an den Sync-Peaks und schneidet Vorlauf ab."""
        if not hasattr(self, 'sync_t_start'):
            print("FEHLER: Führe zuerst align_signals() aus!")
            return

        t_start = self.sync_t_start
        t_end = self.sync_t_end
        csv_time = self.df_csv['Time_Sec']

        # 1. Smarte Maske: Wir nehmen exakt das Fenster 1 bis 6 Sekunden VOR dem ersten Peak
        # und 1 bis 6 Sekunden NACH dem letzten Peak (um aus den Flanken raus zu sein).
        base_mask = ((csv_time >= t_start - 6.0) & (csv_time <= t_start - 1.0)) | \
                    ((csv_time >= t_end + 1.0) & (csv_time <= t_end + 6.0))

        if not base_mask.any():
            base_mask = (csv_time < t_start)  # Fallback

        print(f"Führe Baseline-Korrektur durch (Referenzfenster: {base_mask.sum()} Punkte)...")

        # 2. Offset von allen kinematischen Spalten (inkl. Schläuche) abziehen
        keys = ['True_Lateral', 'True_Longitudinal', 'True_Vertical', 'True_Pitch', 'True_Roll', 'True_Yaw']
        for key in keys:
            offset = np.median(self.df_csv.loc[base_mask, key])
            self.df_csv[key] -= offset
            self.df_csv[f"{key}_upper"] -= offset
            self.df_csv[f"{key}_lower"] -= offset

        # 3. Cropping: Wir schneiden alles ab, was mehr als 5 Sekunden vor dem ersten Peak liegt
        crop_time = t_start - 5.0
        self.df_csv = self.df_csv[self.df_csv['Time_Sec'] >= crop_time].reset_index(drop=True)
        self.df_json = self.df_json[self.df_json['Time_Sec'] >= crop_time].reset_index(drop=True)

        print(f"-> Daten gecroppt. Plot startet nun exakt 5s vor dem ersten Peak.")

    def plot_sync_check(self):
        """Visualisierung aller 6 DoF inkl. Unsicherheits-Schläuchen."""

        def _add_uncertainty_trace(fig, df, key, color, name):
            # 1. Der Unsicherheits-Schlauch (Transparente Fläche)
            fig.add_trace(go.Scatter(
                x=np.concatenate([df['Time_Sec'], df['Time_Sec'][::-1]]),
                y=np.concatenate([df[f"{key}_upper"], df[f"{key}_lower"][::-1]]),
                fill='toself',
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} Uncert."
            ))
            # 2. Die nominelle SURF-Linie (gestrichelt)
            fig.add_trace(go.Scatter(
                x=df['Time_Sec'], y=df[key],
                name=f"SURF {name}",
                line=dict(color=color, width=2, dash='dash')
            ))

        # --- FIGUR 1: TRANSLATIONEN (X, Y, Z) ---
        fig_trans = go.Figure()

        # ETD Daten (Durchgezogen)
        fig_trans.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['lateral'], name="ETD Lateral (X)",
                                       line=dict(color='red')))
        fig_trans.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['longitudinal'], name="ETD Long. (Y)",
                                       line=dict(color='green')))
        fig_trans.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['vertical'], name="ETD Vert. (Z)",
                                       line=dict(color='blue')))

        # SURF Daten (Gestrichelt + Schlauch)
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Lateral', 'rgb(255, 0, 0)', 'Lat (X)')
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Longitudinal', 'rgb(0, 255, 0)', 'Long (Y)')
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Vertical', 'rgb(0, 0, 255)', 'Vert (Z)')

        fig_trans.update_layout(title="6 DoF Check: Translationen", xaxis_title="Zeit [s]", yaxis_title="Position [mm]",
                                hovermode="x unified")
        fig_trans.show()

        # --- FIGUR 2: ROTATIONEN (Pitch, Roll, Yaw) ---
        fig_rot = go.Figure()

        # ETD Daten
        fig_rot.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['pitch'], name="ETD Pitch",
                                     line=dict(color='orange')))
        fig_rot.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['roll'], name="ETD Roll", line=dict(color='purple')))
        fig_rot.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['yaw'], name="ETD Yaw", line=dict(color='brown')))

        # SURF Daten
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Pitch', 'rgb(255, 165, 0)', 'Pitch')
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Roll', 'rgb(128, 0, 128)', 'Roll')
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Yaw', 'rgb(165, 42, 42)', 'Yaw')

        fig_rot.update_layout(title="6 DoF Check: Rotationen", xaxis_title="Zeit [s]", yaxis_title="Winkel [°]",
                              hovermode="x unified")
        fig_rot.show()