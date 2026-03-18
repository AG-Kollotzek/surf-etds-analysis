# surf-etds-analysis
Python-basierte Evaluierungsplattform zur Qualitätssicherung des ExacTrac Surface Scanners (SGRT). Synchronisation, Flankenerkennung und kinematische Analyse von Tracking-Daten (.json) und hardwareseitigen Telemetriedaten (.csv) der SURF-Test-Unit.

Dieses Repository dient der automatisierten Auswertung und dem Vergleich von zwei Datenströmen:

  - SURF-Test-Unit (Ground Truth): Präzise geloggte Roboter-Bewegungen (Linear- und Rotationsachsen) via .csv.

  - ExacTrac (ETD) Scanner: Aufgezeichnete Tracking-Logs des SGRT-Systems via .json.

Kernfunktionen: 
  * Automatisches Time-Alignment beider Datenströme (z.B. über 5mm Referenz-Peaks).

  * Plateau- und Flankenerkennung zur Evaluierung von stationären Endpositionen (Mittelwert & Standardabweichung).

  * Kinematische Transformation der Aktor-Koordinaten in das Patienten-/Couch-Koordinatensystem inkl. Fehlerfortpflanzung.

  * Sicherstellung der QA-Anforderungen gemäß aktueller ESTRO-Guidelines für Oberflächenscanner.
