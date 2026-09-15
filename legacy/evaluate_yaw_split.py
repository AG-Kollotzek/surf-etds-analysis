import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from pathlib import Path
import re


def analyze_yaw_split():
    print("=== STARTE YAW-SPLIT ANALYSE FÜR VARIABLE GESCHWINDIGKEIT ===")

    # Pfad zum übergeordneten Ergebnis-Ordner anpassen!
    base_dir = Path(
        'path/to/SURF/20260310_Messung_4/20260310_messung4/03_Ergebnisse/Variable_Geschwindigkeit')

    # Suche alle generierten QA_Reports in den Unterordnern
    csv_files = list(base_dir.rglob("QA_Report_Messung_*.csv"))

    if not csv_files:
        print(f"FEHLER: Keine CSV-Dateien im Verzeichnis {base_dir} gefunden.")
        return

    print(f"-> {len(csv_files)} Messungen für die Auswertung gefunden.\n")

    # Arrays zum Sammeln der globalen Daten (falls du später einen übergreifenden Plot machen willst)
    summary_data = []

    for csv_file in sorted(csv_files):
        # Messungs-ID aus dem Dateinamen extrahieren (z.B. "19" aus "QA_Report_Messung_19.csv")
        match = re.search(r'Messung_(\d+)', csv_file.stem)
        meas_id = match.group(1) if match else "Unbekannt"

        print(f"--- Werte Messung {meas_id} aus ---")

        # CSV einlesen (comment='#' ignoriert unsere Metadaten-Zeilen am Anfang)
        try:
            df = pd.read_csv(csv_file, comment='#', header=[0, 1])
        except Exception as e:
            print(f"  [!] Fehler beim Lesen der CSV: {e}")
            continue

        # Prüfen, ob die Yaw-Spalten existieren
        if ('Yaw', 'Mean_Diff') not in df.columns:
            print(f"  [!] Keine Yaw-Daten in Messung {meas_id} gefunden. Überspringe...")
            continue

        # Daten extrahieren
        peak_ids = df[('Meta', 'Peak_ID')]
        yaw_mean = df[('Yaw', 'Mean_Diff')]
        yaw_std = df[('Yaw', 'Std_Diff')]

        # Masken für Ungerade (Hin-Fahrt) und Gerade (Rück-Fahrt)
        odd_mask = peak_ids % 2 != 0
        even_mask = peak_ids % 2 == 0

        odd_ids, odd_yaw, odd_std = peak_ids[odd_mask], yaw_mean[odd_mask], yaw_std[odd_mask]
        even_ids, even_yaw, even_std = peak_ids[even_mask], yaw_mean[even_mask], yaw_std[even_mask]

        if len(odd_ids) < 2 or len(even_ids) < 2:
            print(f"  [!] Zu wenige Datenpunkte in Messung {meas_id} für eine Regression.")
            continue

        # Lineare Regressionen
        slope_odd, int_odd, r_odd, p_odd, err_odd = linregress(odd_ids, odd_yaw)
        slope_even, int_even, r_even, p_even, err_even = linregress(even_ids, even_yaw)

        # Plot erstellen
        plt.figure(figsize=(10, 6))

        # Plot Ungerade (Hin-Fahrt)
        plt.errorbar(odd_ids, odd_yaw, yerr=odd_std, fmt='o', capsize=5, markersize=8,
                     label=f'Ungerade (Hin-Fahrt)', color='#1f77b4', ecolor='#1f77b4', elinewidth=2)
        trend_x_odd = np.linspace(min(odd_ids), max(odd_ids), 10)
        plt.plot(trend_x_odd, int_odd + slope_odd * trend_x_odd, '--', color='#1f77b4',
                 label=f'Trend Ungerade (p={p_odd:.2f})')

        # Plot Gerade (Rück-Fahrt)
        plt.errorbar(even_ids, even_yaw, yerr=even_std, fmt='s', capsize=5, markersize=8,
                     label=f'Gerade (Rück-Fahrt)', color='#ff7f0e', ecolor='#ff7f0e', elinewidth=2)
        trend_x_even = np.linspace(min(even_ids), max(even_ids), 10)
        plt.plot(trend_x_even, int_even + slope_even * trend_x_even, '--', color='#ff7f0e',
                 label=f'Trend Gerade (p={p_even:.2f})')

        # Formatierung
        plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        plt.title(f'Yaw-Abweichung (ETD vs. SURF) - Messung {meas_id}\n(Getrennt nach Bewegungsrichtung)', fontsize=14)
        plt.xlabel('Peak ID (Zeitlicher Verlauf)', fontsize=12)
        plt.ylabel('Mittlere Yaw-Differenz [Grad]', fontsize=12)
        plt.xticks(peak_ids)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()

        # Bild direkt im Ordner der jeweiligen Messung speichern
        plot_path = csv_file.parent / f"Yaw_Trend_Split_M{meas_id}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()  # Wichtig, damit der RAM nicht voll wird

        # Konsolen-Output der Statistiken
        mean_odd_val = odd_yaw.mean()
        mean_even_val = even_yaw.mean()
        hysterese = abs(mean_odd_val - mean_even_val)

        print(f"  [✓] Plot gespeichert unter: {plot_path.name}")
        print(f"      Hin-Fahrt  (Ungerade): {mean_odd_val:+.4f}° (Drift p-Wert: {p_odd:.2f})")
        print(f"      Rück-Fahrt (Gerade):   {mean_even_val:+.4f}° (Drift p-Wert: {p_even:.2f})")
        print(f"      --> Mechanisches Umkehrspiel (Hysterese): {hysterese:.4f}°\n")

        # Daten für eine spätere Zusammenfassung speichern
        summary_data.append({
            'Messung': meas_id,
            'Mean_Ungerade': mean_odd_val,
            'Mean_Gerade': mean_even_val,
            'Hysterese': hysterese
        })

    # Optional: Zusammenfassung als DataFrame anzeigen
    if summary_data:
        print("=== ZUSAMMENFASSUNG ALLER MESSUNGEN ===")
        df_summary = pd.DataFrame(summary_data).set_index('Messung')
        print(df_summary)

        # Speichert die Zusammenfassung ebenfalls ab
        summary_path = base_dir / "Zusammenfassung_Hysterese_VariableSpeed.csv"
        df_summary.to_csv(summary_path)
        print(f"\nZusammenfassung gespeichert unter: {summary_path}")


if __name__ == "__main__":
    analyze_yaw_split()