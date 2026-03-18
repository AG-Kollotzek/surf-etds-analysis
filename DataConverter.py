import pandas as pd
import numpy as np
import json
from scipy import signal
import plotly.graph_objects as go


class ETDQAProcessor:
    def __init__(self):
        self.df_csv = None
        self.df_json = None
        self.time_offset = 0

    def load_csv(self, file_path):
        """Lädt das SURF-Terminal Log (Trennzeichen ;)"""
        self.df_csv = pd.read_csv(file_path, sep=';', decimal='.')
        return self.df_csv

    def load_json(self, file_path):
        """Lädt das ETD-Tracking Log und extrahiert shiftValues"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        results = []
        for entry in data['trackingResults']:
            if not entry.get('trackingLost', False):
                # Extrahiere shiftValues (Strings zu Float)
                shifts = {k: float(v) for k, v in entry['shiftValues'].items()}
                shifts['timestamp_ms'] = entry['timestamp']
                results.append(shifts)

        self.df_json = pd.DataFrame(results)
        # Zeit in Sekunden umrechnen (ETD Timestamps sind oft ms ab Start)
        self.df_json['Time_Sec'] = (self.df_json['timestamp_ms'] - self.df_json['timestamp_ms'].iloc[0]) / 1000.0
        return self.df_json

    def align_signals(self):
        """
        Synchronisiert CSV und JSON basierend auf der 5mm H-Bewegung.
        Wir nutzen die Kreuzkorrelation der Gradienten (Änderungsraten).
        """
        # 1. Referenz aus CSV (H-Achse Änderung)
        # Wir nehmen den Absolutwert des Gradienten, um Vorzeichen zu ignorieren
        sig_csv = np.abs(np.gradient(self.df_csv['Pos_H'].values))

        # 2. Signal aus JSON (Magnitude der Verschiebung über alle 3 Achsen)
        # Da H schräg sein kann, nehmen wir die kombinierte Änderung von lat/long/vert
        diff_json = self.df_json[['lateral', 'longitudinal', 'vertical']].diff().fillna(0)
        sig_json = np.sqrt((diff_json ** 2).sum(axis=1))  # Euklidische Magnitude der Änderung

        # Kreuzkorrelation berechnen
        correlation = signal.correlate(sig_csv, sig_json, mode='full')
        lags = signal.correlation_lags(len(sig_csv), len(sig_json), mode='full')

        # Finde den Lag mit der höchsten Übereinstimmung
        best_lag_idx = np.argmax(correlation)
        lag_steps = lags[best_lag_idx]

        # Zeit-Offset berechnen (basierend auf der Abtastrate des CSV, ca. 10Hz?)
        dt = self.df_csv['Time_Sec'].diff().mean()
        self.time_offset = lag_steps * dt

        print(f"Synchronisation abgeschlossen. Berechneter Offset: {self.time_offset:.3f} s")
        return self.time_offset

    def plot_sync_check(self):
        """Visualisierung zur Kontrolle des Alignments"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['Pos_H'], name="SURF Pos_H (Ground Truth)"))

        # JSON Zeit verschieben
        fig.add_trace(go.Scatter(x=self.df_json['Time_Sec'] + self.time_offset,
                                 y=self.df_json['longitudinal'],  # Beispielachse
                                 name="ETD Lateral (Aligned)"))

        fig.update_layout(title="Synchronisations-Check", xaxis_title="Zeit [s]", yaxis_title="Position [mm]")
        fig.show()

# Beispielhafte Anwendung:
# processor = ETDQAProcessor()
# processor.load_csv('dein_file.csv')
# processor.load_json('dein_file.json')
# processor.align_signals()
# processor.plot_sync_check()