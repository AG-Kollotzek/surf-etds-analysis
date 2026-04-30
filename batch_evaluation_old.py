import os
from pathlib import Path
from DataConverter import ETDQAProcessor

# Automatisch aus dem Protokoll extrahierte Zuordnung
# Format: 'Messungs_ID': {'Gruppe': 'Gruppenname', 'ETD': 'JSON_Zeitcode', 'CSV': 'CSV_Zeitcode'}
MEASUREMENT_10032026 = {
    '1': {'Gruppe': 'Standardmessung', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '160617', 'CSV': '160448'},
    '2': {'Gruppe': 'Standardmessung', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '161117', 'CSV': '160950'},
    '3': {'Gruppe': 'Standardmessung', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '161416', 'CSV': '161302'},
    '4': {'Gruppe': 'Longitudinal', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '161724', 'CSV': '161538'},
    '5': {'Gruppe': 'Longitudinal', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '162054', 'CSV': '161859'},
    '6': {'Gruppe': 'Longitudinal', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '162326', 'CSV': '162130'},
    '7': {'Gruppe': 'Vertikal', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '162647', 'CSV': '162451'},
    '8': {'Gruppe': 'Vertikal', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '162922', 'CSV': '162728'},
    '9': {'Gruppe': 'Vertikal', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '163157', 'CSV': '162951'},
    '10': {'Gruppe': 'Rotation', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '163508', 'CSV': '163325'},
    '11': {'Gruppe': 'Rotation', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '163758', 'CSV': '163613'},
    '12': {'Gruppe': 'Rotation', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '164007', 'CSV': '172148'},
    '14': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '173928', 'CSV': '173645'},
    '15': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '174334', 'CSV': '174056'},
    '16': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area' : 'PhantomWithBuffer', 'Heatingpads' : 'OFF', 'ETD': '174832', 'CSV': '174558'},
    '17': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area' : 'Fitting', 'Heatingpads' : 'OFF', 'ETD': '175259', 'CSV': '175019'},
    '18': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area' : 'OnlyFrontSurface', 'Heatingpads' : 'OFF', 'ETD': '175635', 'CSV': '175356'},
    '19': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area' : 'Fiting', 'Heatingpads' : 'OFF', 'ETD': '180108', 'CSV': '175824'},
    '21': {'Gruppe': 'Vertical_Slide', 'Heatingpads': 'OFF', 'ETD': '182337', 'CSV': '182011'},
    '22': {'Gruppe': 'Vertical_Slide', 'Heatingpads': 'OFF', 'ETD': '194758', 'CSV': '193557'},
    '24': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '161724', 'CSV': '161538'},
    '25': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '162054', 'CSV': '161859'},
    '26': {'Gruppe': 'Longitudinal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '162326', 'CSV': '162130'},
    '27': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '162647', 'CSV': '162451'},
    '28': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '162922', 'CSV': '162728'},
    '29': {'Gruppe': 'Vertikal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '163157', 'CSV': '162951'},
    '30': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '163508', 'CSV': '163325'},
    '31': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '163758', 'CSV': '163613'},
    '32': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '164007', 'CSV': '172148'},
    '33': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '173928', 'CSV': '173645'},
    '34': {'Gruppe': 'Vertical_Slide', 'Heatingpads': '32', 'ETD': '193528', 'CSV': '193140'},
    '35': {'Gruppe': 'Vertical_Slide', 'Heatingpads': '32', 'ETD': '193801', 'CSV': '193557'},
}


def main():
    print("=== STARTE AUTOMATISCHE BATCH-EVALUIERUNG ===")

    # BASIS-VERZEICHNIS ANPASSEN!
    data_dir = Path('path/to/SURF/20260310_Messung_4/20260310_messung4')

    # Übergeordneter Ausgabe-Ordner
    results_base_dir = 'path/to/SURF' / "Paper_Ergebnisse"
    results_base_dir.mkdir(parents=True, exist_ok=True)

    # Schleife durch alle im Wörterbuch definierten Messungen
    for meas_id, data in MEASUREMENT_10032026.items():
        gruppe = data['Gruppe']
        etd_code = data['ETD']
        csv_code = data['CSV']

        print(f"\n[{gruppe.upper()}] ---> Verarbeite Messung {meas_id}")

        # Ordnerstruktur anlegen (z.B. 03_Ergebnisse/Rotation/Messung_10/)
        out_dir = results_base_dir / gruppe / f"Messung_{meas_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Finde die Dateien rekursiv (egal in welchem Unterordner sie im Rohdaten-Verzeichnis liegen)
        # rglob durchsucht alle Ordner nach Dateien, die mit dem Code enden
        csv_files = list(data_dir.rglob(f"*{csv_code}.csv"))
        json_files = list(data_dir.rglob(f"*{etd_code}.json"))

        if not csv_files or not json_files:
            print(
                f"   [!] ÜBERSPRUNGEN: Dateien für Messung {meas_id} (CSV: {csv_code}, ETD: {etd_code}) nicht im Ordner gefunden.")
            continue

        # Nimm den ersten Treffer
        csv_path = str(csv_files[0])
        json_path = str(json_files[0])

        processor = ETDQAProcessor(terminal_version='legacy')
        try:
            # 1. Daten laden und aufbereiten
            processor.load_csv(csv_path)
            processor.load_json(json_path)
            processor.apply_kinematics(couch_angle=0.0)

            # 2. Alignment und Nullung
            processor.align_signals()
            processor.apply_baseline_and_crop()

            # 3. Plateaus auswerten (die flexible kinetische Version von vorhin)
            csv_output = out_dir / f"QA_Report_Messung_{meas_id}.csv"
            processor.evaluate_plateaus_and_export(output_file=str(csv_output))

            # 4. NEU: Alle 3 Plots speichern
            processor.export_all_plots(output_dir=str(out_dir), prefix=f"M{meas_id}")
            print(f"   [✓] Messung {meas_id} erfolgreich abgeschlossen!")

        except Exception as e:
            print(f"   [X] FEHLER bei Messung {meas_id}: {e}")


if __name__ == "__main__":
    main()