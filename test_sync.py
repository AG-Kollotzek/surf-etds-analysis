import os
from DataConverter import ETDQAProcessor
from pathlib import Path


def main():
    print("--- SURF ETD QA Pipeline ---")

    # 1. Pfade zu deinen Testdaten definieren
    # HINWEIS: Passe diese Pfade an die Struktur deines Repos an.
    data_dir = Path('path/to/SURF/20260310_Messung_4/20260310_messung4')
    csv_full_path = data_dir / "csv" / "ETD_QA_PoP_SingleCouchOrientation_20260310_182011.csv"
    json_full_path = data_dir / "01_json" / "TrackingResult_2026-03-10_18-23-37.json"

    # --- NEU: INTERAKTIVE ABFRAGE ---
    # Fragt den Nutzer nach dem Typ (z.B. 'Baseline', 'Longitudinal_5mm' oder 'Couch_90')
    messung_typ = input("Welcher Typ von Messung ist das? (Eingabe wird an Dateinamen angehängt): ").strip()

    # Dateiname generieren. Wenn nichts eingegeben wird, Standardname nutzen.
    if messung_typ:
        report_name = f"QA_Evaluation_Report_{messung_typ}.csv"
    else:
        report_name = "QA_Evaluation_Report.csv"

    report_path = data_dir / report_name

    # Prüfen, ob die Dateien existieren, bevor wir starten
    if not os.path.exists(csv_full_path):
        print(f"FEHLER: CSV-Datei nicht gefunden: {csv_full_path}")
        return
    if not os.path.exists(json_full_path):
        print(f"FEHLER: JSON-Datei nicht gefunden: {json_full_path}")
        return

    # 2. Prozessor initialisieren
    processor = ETDQAProcessor(terminal_version='legacy')

    # 3. Daten laden
    print(f"Lade CSV-Daten: {os.path.basename(csv_full_path)}...")
    processor.load_csv(csv_full_path)

    print(f"Lade JSON-Daten: {os.path.basename(json_full_path)}...")
    processor.load_json(json_full_path)

    # Kinematik anwenden (Berechnet True_Lateral, True_Pitch etc.)
    processor.apply_kinematics(couch_angle=0.0)

    # 4. Signale synchronisieren (Time Alignment)
    print("Führe Time-Alignment (Kreuzkorrelation) durch...")
    try:
        processor.align_signals()
    except Exception as e:
        print(f"Fehler beim Alignment: {e}")
        return

    # 5. Baseline-Korrektur und Abschneiden des Leerlaufs
    processor.apply_baseline_and_crop()

    # 6. Statistische Plateau-Erkennung und Export
    # Hier wird nun der dynamisch erstellte Pfad verwendet
    processor.evaluate_plateaus_and_export(output_file=str(report_path))

    # 7. Visuelle Kontrolle
    print("Erstelle Plot zur visuellen Kontrolle...")
    processor.plot_sync_check()


if __name__ == "__main__":
    main()