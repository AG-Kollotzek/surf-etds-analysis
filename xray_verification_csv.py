import pandas as pd
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
# Importiert die Kinematik-Klasse aus deinem Repository
from kinematics_werror import SurfKinematics


def main():
    kin = SurfKinematics()

    # Angenommene Standardunsicherheit (1-Sigma) für eine einzelne ET-Koordinate.
    # Wenn die Unsicherheit der Nachkommastelle als 2-Sigma = 0.1 mm angenommen wird,
    # ist 1-Sigma = 0.05 mm.
    ET_SINGLE_SIGMA = 0.05

    # Chronologische Abfolge der Messpunkte
    raw_sequence = [
        # --- Longitudinal-Block ---
        ('Longitudinal', 0, [0.2, -0.3, 0.2], 'Longitudinal'),
        ('Longitudinal', -10, [0.3, -10.3, 0.2], 'Longitudinal'),
        ('Longitudinal', 0, [0.2, -0.3, 0.2], 'Longitudinal'),

        # --- Vertikal-Block ---
        ('Vertikal', 0, [0.2, -0.3, 0.2], 'Vertikal'),
        ('Vertikal', 10, [0.7, -14.3, -0.2], 'Vertikal'),
        ('Vertikal', -10, [-0.4, 14.1, -0.3], 'Vertikal'),
        ('Vertikal', 0, [0.2, -0.3, 0.2], 'Vertikal'),

        # --- Rotations-Block ---
        ('Rotation', 0, [0.1, -0.3, 0.2], 'Rotation'),
        ('Rotation', 0, [0.2, -0.3, 0.2], 'Rotation'),
        ('Rotation', 5, [0.1, -0.4, 0.2], 'Rotation'),
        ('Rotation', -5, [0.2, -0.2, 0.2], 'Rotation'),
        ('Rotation', 0, [0.2, -0.3, 0.2], 'Rotation'),
        ('Rotation', 30, [-0.2, -0.7, 0.2], 'Rotation'),
        ('Rotation', -30, [0.6, -0.1, 0.2], 'Rotation'),
        ('Rotation', 45, [-0.3, -1.1, 0.2], 'Rotation'),
        ('Rotation', 90, [-0.1, -1.8, 0.2], 'Rotation'),
        ('Rotation', 60, [-0.3, -1.4, 0.2], 'Rotation')
    ]

    # 1. Absolute Werte sammeln (inklusive ufloat für die Theorie)
    absolute_results = []
    for idx, (axis, pos, meas, block) in enumerate(raw_sequence):
        l_val, v_val, r_val = 0.0, 0.0, 0.0
        if axis == 'Longitudinal':
            l_val = float(pos)
        elif axis == 'Vertikal':
            v_val = float(pos)
        elif axis == 'Rotation':
            r_val = float(pos)

        # Berechne die Positionen über die Kinematik (liefert ufloat-Werte)
        calc_dict = kin.calculate_task_space(l_val, v_val, r_val, couch_angle_raw_deg=0.0)

        # Behandle die ExacTrac-Messungen als ufloats mit dem definierten Fehler
        m_lat = ufloat(meas[0], ET_SINGLE_SIGMA)
        m_long = ufloat(meas[1], ET_SINGLE_SIGMA)
        m_vert = ufloat(meas[2], ET_SINGLE_SIGMA)

        absolute_results.append({
            'ID': idx, 'Achse': axis, 'Position': pos, 'Block': block,
            'C_Lat': calc_dict['True_Lateral'],
            'C_Long': calc_dict['True_Longitudinal'],
            'C_Vert': calc_dict['True_Vertical'],
            'M_Lat': m_lat, 'M_Long': m_long, 'M_Vert': m_vert
        })

    # Finden der Null-Referenz-Zeile für jeden Block
    block_baselines = {}
    for res in absolute_results:
        if res['Block'] not in block_baselines:
            block_baselines[res['Block']] = res

    # 2. Relative Differenzen und Fehlerfortpflanzung berechnen
    final_table = []
    for res in absolute_results:
        base = block_baselines[res['Block']]

        # Relative Theorie (Subtraktion der ufloats pflanzt Fehler automatisch fort)
        rel_c_lat = res['C_Lat'] - base['C_Lat']
        rel_c_long = res['C_Long'] - base['C_Long']
        rel_c_vert = res['C_Vert'] - base['C_Vert']

        # Relative ET-Messung
        rel_m_lat = res['M_Lat'] - base['M_Lat']
        rel_m_long = res['M_Long'] - base['M_Long']
        rel_m_vert = res['M_Vert'] - base['M_Vert']

        # Gesamtdiskrepanz (Theorie_rel - ET_rel)
        diff_lat = rel_c_lat - rel_m_lat
        diff_long = rel_c_long - rel_m_long
        diff_vert = rel_c_vert - rel_m_vert

        final_table.append({
            'Messpunkt_ID': res['ID'] + 1,
            'Achse': res['Achse'],
            'Position': res['Position'],

            # Trennung in Nominalwert (n) und Standardabweichung (s) für eine saubere Tabelle
            'Rel_Theorie_X (mm)': round(rel_c_lat.n, 3),
            '±_Theorie_X': round(rel_c_lat.s, 3),
            'Rel_Theorie_Y (mm)': round(rel_c_long.n, 3),
            '±_Theorie_Y': round(rel_c_long.s, 3),
            'Rel_Theorie_Z (mm)': round(rel_c_vert.n, 3),
            '±_Theorie_Z': round(rel_c_vert.s, 3),

            'Rel_ExacTrac_X (mm)': round(rel_m_lat.n, 3),
            '±_ExacTrac_X': round(rel_m_lat.s, 3),
            'Rel_ExacTrac_Y (mm)': round(rel_m_long.n, 3),
            '±_ExacTrac_Y': round(rel_m_long.s, 3),
            'Rel_ExacTrac_Z (mm)': round(rel_m_vert.n, 3),
            '±_ExacTrac_Z': round(rel_m_vert.s, 3),

            'Diskrepanz_dX (mm)': round(diff_lat.n, 3),
            '±_Delta_X': round(diff_lat.s, 3),
            'Diskrepanz_dY (mm)': round(diff_long.n, 3),
            '±_Delta_Y': round(diff_long.s, 3),
            'Diskrepanz_dZ (mm)': round(diff_vert.n, 3),
            '±_Delta_Z': round(diff_vert.s, 3)
        })

    df_final = pd.DataFrame(final_table)
    df_final.to_csv("xray_verification_with_errors.csv", index=False)
    print("\nKalibriertabelle inklusive Fehlerfortpflanzung wurde als 'xray_verification_with_errors.csv' gespeichert.")

    # Kompakte Konsolenvorschau der wichtigsten Abweichungsspalten
    print("\nErgebnisübersicht (Diskrepanzen mit Unsicherheitsintervallen):")
    preview_cols = ['Messpunkt_ID', 'Achse', 'Position', 'Diskrepanz_dX (mm)', '±_Delta_X', 'Diskrepanz_dY (mm)',
                    '±_Delta_Y', 'Diskrepanz_dZ (mm)', '±_Delta_Z']
    print(df_final[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()