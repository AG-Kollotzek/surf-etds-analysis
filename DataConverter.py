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
        """Berechnet die klinischen Koordinaten und speichert sie im DataFrame."""
        print("Berechne kinematische Transformation in klinische Koordinaten...")
        kin = SurfKinematics()

        # 1. Raw-Daten holen
        h_raw = self.df_csv['Pos_H'].values
        v_raw = self.df_csv['Pos_V'].values
        r_raw = self.df_csv['Pos_R'].values

        # 2. Berechnung inkl. Fehlerfortpflanzung durchführen
        res = kin.calculate_task_space(h_raw, v_raw, r_raw, couch_angle)

        # 3. WICHTIG: ufloats entpacken! Pandas & Plotly brauchen reine Floats.
        # unp.nominal_values holt den reinen Zahlenwert.
        self.df_csv['True_Lateral'] = unp.nominal_values(res['True_Lateral'])
        self.df_csv['True_Longitudinal'] = unp.nominal_values(res['True_Longitudinal'])
        self.df_csv['True_Vertical'] = unp.nominal_values(res['True_Vertical'])
        self.df_csv['True_Pitch'] = unp.nominal_values(res['True_Pitch'])
        self.df_csv['True_Roll'] = unp.nominal_values(res['True_Roll'])
        self.df_csv['True_Yaw'] = unp.nominal_values(res['True_Yaw'])

        # (Optional) Du könntest hier auch die Fehler für Error-Bars speichern:
        # self.df_csv['Err_Lateral'] = unp.std_devs(res['True_Lateral'])

    def align_signals(self, sync_axis='Pos_H', threshold=4.5):
        """
        Implementiert die MATLAB-Sync-Logik:
        1. Findet Mitten des ersten und letzten Peaks.
        2. Berechnet Offset UND Skalierungsfaktor für Clock-Drift Korrektur.
        """

        def get_peak_midpoints(times, values, thresh):
            # Debouncing/Smoothing wie in MATLAB (movmedian 5)
            v_smooth = pd.Series(values).rolling(window=5, center=True).median().fillna(0).values
            mask = np.abs(v_smooth) > thresh

            # Finde Kanten (Edges)
            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0] - 1

            if len(starts) < 2:
                return None, None

            # Mitten des ersten und letzten Peaks berechnen
            # (Index-Mitte und dann die entsprechende Zeit holen)
            t_first = times.iloc[int(round((starts[0] + ends[0]) / 2))]
            t_last = times.iloc[int(round((starts[-1] + ends[-1]) / 2))]

            return t_first, t_last

        # 1. Peak-Mitten finden
        csv_first, csv_last = get_peak_midpoints(self.df_csv['Time_Sec'], self.df_csv[sync_axis], threshold)
        json_first, json_last = get_peak_midpoints(self.df_json['Time_Sec'], self.df_json['Vector_Mag'], threshold)

        if csv_first is None or json_first is None:
            raise ValueError("Nicht genügend 5mm Peaks (erster & letzter) für Sync gefunden.")

        # 2. Delta Zeiten berechnen
        delta_t_csv = csv_last - csv_first
        delta_t_json = json_last - json_first

        # 3. Skalierungsfaktor berechnen (Clock Drift Kompensation)
        scale_factor = delta_t_csv / delta_t_json

        # 4. Die JSON Zeitachse permanent transformieren
        # Wir setzen den ersten CSV-Peak als Nullpunkt und skalieren von dort aus
        self.df_json['Time_Sec'] = (self.df_json['Time_Sec'] - json_first) * scale_factor + csv_first

        # Der Offset ist nun 0, da wir die Spalte direkt transformiert haben
        self.time_offset = 0

        print(f"Sync erfolgreich:")
        print(f" -> Skalierungsfaktor: {scale_factor:.6f}")
        print(f" -> Referenz-Peak bei: {csv_first:.2f} s")

        return scale_factor

    def plot_sync_check(self):
        """
        Visualisierung der synchronisierten Daten.
        WICHTIG: Da die Zeitachse in align_signals() bereits transformiert wurde,
        nutzen wir hier direkt self.df_json['Time_Sec'] ohne extra Offset.
        """

        # --- PLOT 1: TRANSLATIONEN (Linear Shifts) ---
        fig1 = go.Figure()

        # ETD Rohdaten (bereits synchronisiert)
        fig1.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['lateral'],
                                  name="ETD Lateral (X)", line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['longitudinal'],
                                  name="ETD Longitudinal (Y)", line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['vertical'],
                                  name="ETD Vertical (Z)", line=dict(color='blue')))

        # SURF Ground Truth (JETZT KLINISCHE KOORDINATEN)
        fig1.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['True_Lateral'],
                                  name="SURF True Lateral (X)", line=dict(color='darkred', width=2, dash='dash')))
        fig1.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['True_Longitudinal'],
                                  name="SURF True Long. (Y)", line=dict(color='darkgreen', width=2, dash='dash')))
        fig1.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['True_Vertical'],
                                  name="SURF True Vertical (Z)", line=dict(color='darkblue', width=2, dash='dash')))

        fig1.update_layout(
            title=f"Synchronisations-Check: Translationen (Mode: {self.terminal_version})",
            xaxis_title="Zeit [s]",
            yaxis_title="Position [mm]",
            hovermode="x unified"
        )
        fig1.show()

        # --- PLOT 2: ROTATIONEN (Angular Shifts) ---
        fig2 = go.Figure()

        # ETD Rotationen (bereits synchronisiert)
        fig2.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['pitch'],
                                  name="ETD Pitch", line=dict(color='orange')))
        fig2.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['roll'],
                                  name="ETD Roll", line=dict(color='purple')))
        fig2.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['yaw'],
                                  name="ETD Yaw", line=dict(color='brown')))


        # SURF Ground Truth (JETZT KLINISCHE KOORDINATEN)
        fig2.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['True_Pitch'],
                                  name="SURF True Pitch", line=dict(color='darkorange', width=2, dash='dash')))
        fig2.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['True_Roll'],
                                  name="SURF True Roll", line=dict(color='purple', width=2, dash='dash')))
        fig2.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['True_Yaw'],
                                  name="SURF True Yaw", line=dict(color='black', width=2, dash='dash')))

        fig2.update_layout(
            title=f"Synchronisations-Check: Rotationen (Mode: {self.terminal_version})",
            xaxis_title="Zeit [s]",
            yaxis_title="Winkel [°]",
            hovermode="x unified"
        )
        fig2.show()