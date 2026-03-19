import os
from DataConverter import ETDQAProcessor
from pathlib import Path

def main():
    print("Starte ETD QA Synchronisations-Test...")

    # 1. Pfade zu deinen Testdaten definieren
    # HINWEIS: Passe diese Pfade an die Struktur deines Repos an.
    # Hier nutze ich die Namen der Dateien, die du hochgeladen hast.
    # (Achtung: Für ein echtes Alignment sollten CSV und JSON natürlich aus derselben Messung stammen!)
    data_dir = Path(r'path\to\SURF\20260310_Messung_4\20260310_messung4')
    csv_full_path = data_dir / "csv" / "ETD_QA_PoP_SingleCouchOrientation_20260310_182433.csv"
    json_full_path = data_dir / "01_json" / "TrackingResult_2026-03-10_18-26-30.json"

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
    print("CSV erfolgreich geladen.")

    print(f"Lade JSON-Daten: {os.path.basename(json_full_path)}...")
    processor.load_json(json_full_path)
    print("JSON erfolgreich geladen.")

    #Kinematik anwenden
    processor.apply_kinematics(couch_angle=0.0)

    # 4. Signale synchronisieren (Time Alignment)
    print("Führe Time-Alignment (Kreuzkorrelation) durch...")
    try:
        time_offset = processor.align_signals()
        print(f"Erfolg! Berechneter Zeitversatz: {time_offset:.4f} Sekunden.")
    except Exception as e:
        print(f"Fehler beim Alignment: {e}")
        return

    # 5. Visuelle Kontrolle (öffnet einen interaktiven Graphen im Browser)
    print("Erstelle Plot zur visuellen Kontrolle...")
    processor.plot_sync_check()

if __name__ == "__main__":
    main()