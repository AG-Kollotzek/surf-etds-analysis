import os
import sys
import datetime
from pathlib import Path
from DataConverter import ETDQAProcessor


# --- NEU: Der Logger, der sich in den Print-Befehl einklinkt ---
class DualLogger(object):
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        # Wir öffnen die Log-Datei im Modus 'a' (append) oder 'w' (write)
        self.log = open(log_filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Automatisch aus dem Protokoll extrahierte Zuordnung
MEASUREMENTS = {
    '1': {'Gruppe': 'Standardmessung', 'ETD': '160617', 'CSV': '160448'},
    '2': {'Gruppe': 'Standardmessung', 'ETD': '161117', 'CSV': '160950'},
    '3': {'Gruppe': 'Standardmessung', 'ETD': '161416', 'CSV': '161302'},
    '4': {'Gruppe': 'Longitudinal', 'ETD': '161724', 'CSV': '161538'},
    '5': {'Gruppe': 'Longitudinal', 'ETD': '162054', 'CSV': '161859'},
    '6': {'Gruppe': 'Longitudinal', 'ETD': '162326', 'CSV': '162130'},
    '7': {'Gruppe': 'Vertikal', 'ETD': '162647', 'CSV': '162451'},
    '8': {'Gruppe': 'Vertikal', 'ETD': '162922', 'CSV': '162728'},
    '9': {'Gruppe': 'Vertikal', 'ETD': '163157', 'CSV': '162951'},
    '10': {'Gruppe': 'Rotation', 'ETD': '163508', 'CSV': '163325'},
    '11': {'Gruppe': 'Rotation', 'ETD': '163758', 'CSV': '163613'},
    '12': {'Gruppe': 'Rotation', 'ETD': '164007', 'CSV': '172148'},
    '14': {'Gruppe': 'Variable_Geschwindigkeit', 'ETD': '173928', 'CSV': '173645'},
    '15': {'Gruppe': 'Variable_Geschwindigkeit', 'ETD': '174334', 'CSV': '174056'},
    '16': {'Gruppe': 'Variable_Geschwindigkeit', 'ETD': '174832', 'CSV': '174558'},
    '17': {'Gruppe': 'Variable_Geschwindigkeit', 'ETD': '175259', 'CSV': '175019'},
    '18': {'Gruppe': 'Variable_Geschwindigkeit', 'ETD': '175635', 'CSV': '175356'},
    '19': {'Gruppe': 'Variable_Geschwindigkeit', 'ETD': '181427', 'CSV': '175824'},
    '21': {'Gruppe': 'Vertical_Slide', 'ETD': '182337', 'CSV': '182011'},
    '22': {'Gruppe': 'Vertical_Slide', 'ETD': '194758', 'CSV': '193557'},
    '33': {'Gruppe': 'Variable_Geschwindigkeit_Rotation', 'ETD': '192853', 'CSV': '192615'},
    '34': {'Gruppe': 'Vertical_Slide', 'ETD': '193528', 'CSV': '193140'},
    '35': {'Gruppe': 'Vertical_Slide', 'ETD': '193801', 'CSV': '193557'},
    # Fehlende hier einfach ergänzen...
}


def main():
    # BASIS-VERZEICHNIS ANPASSEN!
    data_dir = Path('path/to/SURF/20260310_Messung_4/20260310_messung4')

    # Übergeordneter Ausgabe-Ordner
    results_base_dir = data_dir / "03_Ergebnisse"
    results_base_dir.mkdir(parents=True, exist_ok=True)

    # --- LOGGING INITIALISIEREN ---
    log_filename = results_base_dir / f"Batch_Run_Log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = DualLogger(str(log_filename))  # Alle Prints abfangen!

    print("==================================================")
    print(f"STARTE AUTOMATISCHE BATCH-EVALUIERUNG")
    print(f"Zeitpunkt: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================\n")



    processor = ETDQAProcessor(terminal_version='legacy')
    try:
        # 1. Daten laden und aufbereiten
        processor.load_csv(csv_path)
        processor.load_json(json_path)
        processor.apply_kinematics(couch_angle=0.0)

        # 2. Alignment und Nullung
        processor.align_signals()
        processor.apply_baseline_and_crop()

        # 3. Plateaus auswerten
        csv_output = out_dir / f"QA_Report_Messung_{meas_id}.csv"
        processor.evaluate_plateaus_and_export(output_file=str(csv_output))

        # 4. Plots speichern
        processor.export_all_plots(output_dir=str(out_dir), prefix=f"M{meas_id}")
        print(f"   [✓] SUCCESS: Messung {meas_id} erfolgreich verarbeitet und gespeichert.")

    except Exception as e:
        # Das fängt jeden Fehler (Code-Crash, Division durch 0 etc.) auf und schreibt ihn ins Log
        print(f"   [X] CRITICAL ERROR bei Messung {meas_id}: {e}")
        import traceback
        print(traceback.format_exc())  # Druckt die genaue Zeile des Fehlers für leichtes Debugging




if __name__ == "__main__":
    main()