import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

try:
    from kinematics import SurfKinematics
except ImportError:
    print("Fehler: Konnte 'kinematics.py' nicht finden. Bitte im selben Ordner ausführen.")
    exit()


class KinematicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SURF QA - Quick Calculator")
        self.root.geometry("800x600")
        self.kin = SurfKinematics()

        # Styling für Übersichtlichkeit
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Unit.TLabel", font=("Segoe UI", 10, "italic"), foreground="gray")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(padx=10, pady=10, expand=True, fill="both")

        self.tab_fwd = ttk.Frame(self.notebook)
        self.tab_inv = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_fwd, text="VORWÄRTS: Achsen -> Phantom")
        self.notebook.add(self.tab_inv, text="RÜCKWÄRTS: Phantom -> Achsen")

        self._setup_fwd_tab()
        self._setup_inv_tab()

    def _setup_fwd_tab(self):
        container = ttk.Frame(self.tab_fwd)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        in_frame = ttk.LabelFrame(container, text="Motor-Parameter")
        in_frame.pack(side="left", fill="y", padx=10)

        self.entries_fwd = {}
        for i, (label, key) in enumerate([("H (mm)", "h"), ("V (mm)", "v"), ("R (°)", "r"), ("Couch (°)", "c")]):
            ttk.Label(in_frame, text=label).grid(row=i, column=0, padx=5, pady=8, sticky="w")
            ent = ttk.Entry(in_frame, width=12)
            ent.insert(0, "0")
            ent.grid(row=i, column=1, padx=5, pady=8)
            self.entries_fwd[key] = ent

        ttk.Button(in_frame, text="Position berechnen", command=self.run_fwd).grid(row=4, columnspan=2, pady=20)

        out_frame = ttk.LabelFrame(container, text="Klinische Ist-Position")
        out_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.res_labels_fwd = {}
        fields = [("Lateral (X)", 'True_Lateral', 'mm'),
                  ("Longitudinal (Y)", 'True_Longitudinal', 'mm'),
                  ("Vertical (Z)", 'True_Vertical', 'mm'),
                  ("Pitch", 'True_Pitch', '°'),
                  ("Roll", 'True_Roll', '°'),
                  ("Yaw", 'True_Yaw', '°')]

        for label_text, key, unit in fields:
            f_frame = ttk.Frame(out_frame)
            f_frame.pack(fill="x", padx=20, pady=5)
            ttk.Label(f_frame, text=f"{label_text}:").pack(side="left")
            lbl = ttk.Label(f_frame, text=f"-- {unit}", style="Big.TLabel")
            lbl.pack(side="right")
            self.res_labels_fwd[key] = (lbl, unit)

    def _setup_inv_tab(self):
        container = ttk.Frame(self.tab_inv)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        in_frame = ttk.LabelFrame(container, text="Zielvorgaben (Felder leer lassen = egal)")
        in_frame.pack(side="left", fill="y", padx=10)

        self.entries_inv = {}
        fields = [("Lat (X)", "x"), ("Long (Y)", "y"), ("Vert (Z)", "z"), ("Pitch (°)", "pitch"),
                  ("Couch (°)", "couch")]
        for i, (label, key) in enumerate(fields):
            ttk.Label(in_frame, text=label).grid(row=i, column=0, padx=5, pady=8, sticky="w")
            ent = ttk.Entry(in_frame, width=12)
            ent.grid(row=i, column=1, padx=5, pady=8)
            self.entries_inv[key] = ent

        ttk.Button(in_frame, text="Optimale Achsen finden", command=self.run_inv).grid(row=5, columnspan=2, pady=20)

        out_frame = ttk.LabelFrame(container, text="Benötigte Motorstellung")
        out_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.res_h = ttk.Label(out_frame, text="H: -- mm", style="Big.TLabel")
        self.res_h.pack(pady=20)
        self.res_v = ttk.Label(out_frame, text="V: -- mm", style="Big.TLabel")
        self.res_v.pack(pady=20)
        ttk.Label(out_frame, text="(Strategie: Minimale Auslenkung)", style="Unit.TLabel").pack(side="bottom", pady=10)

    def run_fwd(self):
        try:
            h = float(self.entries_fwd['h'].get().replace(',', '.'))
            v = float(self.entries_fwd['v'].get().replace(',', '.'))
            r = float(self.entries_fwd['r'].get().replace(',', '.'))
            c = float(self.entries_fwd['c'].get().replace(',', '.'))

            res = self.kin.calculate_task_space(h, v, r, c)

            for key, (lbl, unit) in self.res_labels_fwd.items():
                val = res[key]
                lbl.config(text=f"{val:8.3f} {unit}", foreground="#2c3e50")
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Zahlen eingeben.")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def run_inv(self):
        try:
            targets = {}
            for k in ['x', 'y', 'z', 'pitch']:
                v = self.entries_inv[k].get().strip().replace(',', '.')
                targets[k] = float(v) if v else None

            c_val = self.entries_inv['couch'].get().strip().replace(',', '.')
            couch = float(c_val) if c_val else 0.0

            # Warnung: Couch 0 und X-Verschiebung
            if abs(couch) < 0.1 and targets.get('x') and abs(targets['x']) > 0.5:
                messagebox.showwarning("Geometrie-Warnung",
                                       "Bei Couch 0° kann das Phantom physikalisch nicht seitlich (X) verschoben werden!")
                return

            res = self.kin.find_optimal_axes(targets, couch)

            self.res_h.config(text=f"H: {res['H']:.3f} mm", foreground="#2980b9")
            self.res_v.config(text=f"V: {res['V']:.3f} mm", foreground="#2980b9")

        except ValueError as ve:
            messagebox.showwarning("Nicht möglich", str(ve))
        except Exception as e:
            messagebox.showerror("Fehler", f"Ungültige Eingabe: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    KinematicsApp(root)
    root.mainloop()