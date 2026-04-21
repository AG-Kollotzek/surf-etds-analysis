import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from uncertainties import unumpy as unp

try:
    from kinematics_werror import SurfKinematics
except ImportError:
    print("Fehler: Konnte 'Kinematics.py' nicht finden. Bitte im selben Ordner ausführen.")
    exit()


class KinematicsCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SURF Kinematics Calculator")
        self.root.geometry("500x400")
        self.root.resizable(False, False)

        # Kinematik-Instanz laden
        self.kin_model = SurfKinematics()

        self._build_gui()

    def _build_gui(self):
        # --- Eingabe-Bereich ---
        frame_in = ttk.LabelFrame(self.root, text="Eingabewerte (Motor-Achsen)")
        frame_in.pack(padx=10, pady=10, fill="x")

        # Labels und Eingabefelder definieren
        ttk.Label(frame_in, text="H (mm):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.entry_h = ttk.Entry(frame_in, width=15)
        self.entry_h.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(frame_in, text="V (mm):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.entry_v = ttk.Entry(frame_in, width=15)
        self.entry_v.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(frame_in, text="R (°):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.entry_r = ttk.Entry(frame_in, width=15)
        self.entry_r.grid(row=2, column=1, padx=10, pady=5)

        ttk.Label(frame_in, text="Couch C (°):").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.entry_c = ttk.Entry(frame_in, width=15)
        self.entry_c.grid(row=3, column=1, padx=10, pady=5)

        # --- Berechnen Button ---
        calc_btn = ttk.Button(self.root, text="Klinische Koordinaten berechnen", command=self.calculate)
        calc_btn.pack(pady=5)

        # --- Ausgabe-Bereich (Tabelle) ---
        frame_out = ttk.LabelFrame(self.root, text="Ergebnis (Klinisches Phantom-System)")
        frame_out.pack(padx=10, pady=20, fill="both", expand=True)

        columns = ("Name", "Wert", "Fehler")
        self.tree = ttk.Treeview(frame_out, columns=columns, show="headings", height=6)

        # Spalten formatieren
        self.tree.heading("Name", text="Achse")
        self.tree.column("Name", width=120, anchor="w")

        self.tree.heading("Wert", text="Wert")
        self.tree.column("Wert", width=100, anchor="e")

        self.tree.heading("Fehler", text="Fehler (1σ)")
        self.tree.column("Fehler", width=120, anchor="e")

        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

    def calculate(self):
        # 1. Eingaben auslesen (leere Felder als 0.0 interpretieren)
        def parse_input(entry_widget):
            val = entry_widget.get().strip()
            if not val:
                return 0.0
            return float(val.replace(',', '.'))

        try:
            h_val = parse_input(self.entry_h)
            v_val = parse_input(self.entry_v)
            r_val = parse_input(self.entry_r)
            c_val = parse_input(self.entry_c)
        except ValueError:
            messagebox.showerror("Eingabefehler", "Bitte nur gültige Zahlen eingeben (z.B. 12.5).")
            return

        # 2. Berechnung durchführen
        try:
            # Kinematics erwartet Arrays (aufgrund von unp.uarray), wir übergeben Arrays der Länge 1
            res = self.kin_model.calculate_task_space(
                np.array([h_val]),
                np.array([v_val]),
                np.array([r_val]),
                couch_angle_raw_deg=c_val
            )
        except Exception as e:
            messagebox.showerror("Berechnungsfehler", f"Fehler in Kinematics.py:\n{str(e)}")
            return

        # 3. Tabelle leeren
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 4. Ergebnisse mappen und eintragen
        display_mapping = [
            ('True_Lateral', 'Lateral (X)', 'mm'),
            ('True_Longitudinal', 'Longitudinal (Y)', 'mm'),
            ('True_Vertical', 'Vertical (Z)', 'mm'),
            ('True_Pitch', 'Pitch', '°'),
            ('True_Roll', 'Roll', '°'),
            ('True_Yaw', 'Yaw', '°')
        ]

        for key, name, unit in display_mapping:
            # res[key] ist ein Array mit einem Element vom Typ uncertainties.ufloat
            ufloat_val = res[key][0]
            nominal = ufloat_val.n
            std_dev = ufloat_val.s

            # Werte auf 3 Nachkommastellen runden
            wert_str = f"{nominal:.3f} {unit}"
            fehler_str = f"± {std_dev:.3f} {unit}"

            self.tree.insert("", "end", values=(name, wert_str, fehler_str))


if __name__ == "__main__":
    root = tk.Tk()
    app = KinematicsCalculatorApp(root)
    root.mainloop()