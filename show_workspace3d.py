import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from kinematics import SurfKinematics


def calculate_pure_translation_limits(kin_model, limits, tolerance=0.25):
    """
    Sucht blitzschnell im H/V Aktionsraum nach rein isolierten Translationen.
    """
    # Grid auf 400x400 reduziert -> 160.000 Punkte statt 2.25 Millionen.
    # Mit purem Numpy rechnet das in wenigen Millisekunden.
    h_dense = np.linspace(limits['h'][0], limits['h'][1], 400)
    v_dense = np.linspace(limits['v'][0], limits['v'][1], 400)
    H, V = np.meshgrid(h_dense, v_dense)

    H_flat = H.flatten()
    V_flat = V.flatten()
    R_flat = np.zeros_like(H_flat)

    res = kin_model.calculate_task_space(H_flat, V_flat, R_flat, couch_angle_raw_deg=0.0)

    # Da kein 'uncertainties' mehr genutzt wird, können wir die Arrays direkt nehmen
    y_vals = res['True_Longitudinal']
    z_vals = res['True_Vertical']

    pure_limits = {}

    # --- 1. Reine Z-Translation ---
    mask_y_zero = np.abs(y_vals) < tolerance
    if np.any(mask_y_zero):
        pure_limits['Z_min'] = np.min(z_vals[mask_y_zero])
        pure_limits['Z_max'] = np.max(z_vals[mask_y_zero])
    else:
        pure_limits['Z_min'], pure_limits['Z_max'] = 0, 0

    # --- 2. Reine Y-Translation ---
    mask_z_zero = np.abs(z_vals) < tolerance
    if np.any(mask_z_zero):
        pure_limits['Y_min'] = np.min(y_vals[mask_z_zero])
        pure_limits['Y_max'] = np.max(y_vals[mask_z_zero])
    else:
        pure_limits['Y_min'], pure_limits['Y_max'] = 0, 0

    # --- 3. Reine X-Translation ---
    pure_limits['X_min'] = pure_limits['Y_min']
    pure_limits['X_max'] = pure_limits['Y_max']

    print("\n--- ISOLIERTE LINEARE MAXIMALWEGE (bei Nullstellung der restlichen Achsen) ---")
    print(
        f"Reines X (Lateral):       {pure_limits['X_min']:>6.2f} mm  bis  {pure_limits['X_max']:>6.2f} mm   (Spanne: {pure_limits['X_max'] - pure_limits['X_min']:.2f} mm)")
    print(
        f"Reines Y (Longitudinal):  {pure_limits['Y_min']:>6.2f} mm  bis  {pure_limits['Y_max']:>6.2f} mm   (Spanne: {pure_limits['Y_max'] - pure_limits['Y_min']:.2f} mm)")
    print(
        f"Reines Z (Vertical):      {pure_limits['Z_min']:>6.2f} mm  bis  {pure_limits['Z_max']:>6.2f} mm   (Spanne: {pure_limits['Z_max'] - pure_limits['Z_min']:.2f} mm)")
    print("----------------------------------------------------------------------------------\n")

    return pure_limits


def plot_workspace(kin_model, limits, couch_angles=[0.0, 45.0, 90.0], points_per_axis=10):
    pure_lims = calculate_pure_translation_limits(kin_model, limits)

    h_vals = np.linspace(limits['h'][0], limits['h'][1], points_per_axis)
    v_vals = np.linspace(limits['v'][0], limits['v'][1], points_per_axis)
    r_vals = np.linspace(limits['r'][0], limits['r'][1], points_per_axis)

    H, V, R = np.meshgrid(h_vals, v_vals, r_vals)
    H_flat, V_flat, R_flat = H.flatten(), V.flatten(), R.flatten()

    fig_trans = go.Figure()
    fig_rot = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    all_p, all_r, all_yaw = [], [], []

    for i, angle in enumerate(couch_angles):
        res = kin_model.calculate_task_space(H_flat, V_flat, R_flat, couch_angle_raw_deg=angle)

        # Direkter Array-Zugriff ohne unp.nominal_values
        x = res['True_Lateral']
        y = res['True_Longitudinal']
        z = res['True_Vertical']
        pitch = res['True_Pitch']
        roll = res['True_Roll']
        yaw = res['True_Yaw']

        all_p.extend(pitch);
        all_r.extend(roll);
        all_yaw.extend(yaw)

        fig_trans.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=2.5, color=colors[i % len(colors)], opacity=0.15),
            name=f"Couch {angle}°"
        ))
        fig_rot.add_trace(go.Scatter3d(
            x=pitch, y=roll, z=yaw, mode='markers',
            marker=dict(size=2.5, color=colors[i % len(colors)], opacity=0.15),
            name=f"Couch {angle}°"
        ))

    # --- Fadenkreuze ---
    def add_span_line(fig, x0, x1, y0, y1, z0, z1, name, span_val, unit):
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines+text',
            line=dict(color='black', width=5, dash='solid'),
            text=["", f"Isoliertes Δ {name}: {span_val:.1f} {unit}"],
            textposition="top center",
            textfont=dict(size=13, color='black'),
            name=f"Reine Spanne {name}",
            hoverinfo='skip'
        ))

    cx, cy, cz = 0, 0, 0

    add_span_line(fig_trans, pure_lims['X_min'], pure_lims['X_max'], cy, cy, cz, cz, "X",
                  pure_lims['X_max'] - pure_lims['X_min'], "mm")
    add_span_line(fig_trans, cx, cx, pure_lims['Y_min'], pure_lims['Y_max'], cz, cz, "Y",
                  pure_lims['Y_max'] - pure_lims['Y_min'], "mm")
    add_span_line(fig_trans, cx, cx, cy, cy, pure_lims['Z_min'], pure_lims['Z_max'], "Z",
                  pure_lims['Z_max'] - pure_lims['Z_min'], "mm")

    min_p, max_p = np.min(all_p), np.max(all_p)
    min_r, max_r = np.min(all_r), np.max(all_r)
    min_yaw, max_yaw = np.min(all_yaw), np.max(all_yaw)

    cp, cr, cyaw = 0, 0, 0

    add_span_line(fig_rot, min_p, max_p, cr, cr, cyaw, cyaw, "Pitch", max_p - min_p, "°")
    add_span_line(fig_rot, cp, cp, min_r, max_r, cyaw, cyaw, "Roll", max_r - min_r, "°")
    add_span_line(fig_rot, cp, cp, cr, cr, min_yaw, max_yaw, "Yaw", max_yaw - min_yaw, "°")

    fig_trans.update_layout(
        title="ETD Phantom: Translations-Aktionsraum (Idealisiert)",
        scene=dict(xaxis_title='Lateral (X) [mm]', yaxis_title='Longitudinal (Y) [mm]',
                   zaxis_title='Vertical (Z) [mm]'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig_rot.update_layout(
        title="ETD Phantom: Rotations-Aktionsraum (Idealisiert)",
        scene=dict(xaxis_title='Pitch [°]', yaxis_title='Roll [°]', zaxis_title='Yaw [°]'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig_trans.show()
    fig_rot.show()


def plot_z_pitch_behavior(kin_model, limits, y_target=0.0):
    """
    Berechnet die Kopplung von Z und Pitch, während Y auf einem festen Wert gehalten wird.
    """
    # 1. Hochauflösender Scan der physischen V-Achse (bestimmt primär den Pitch)
    v_vals = np.linspace(limits['v'][0], limits['v'][1], 1000)

    # 2. Hilfswerte aus der Geometrie (intern in Kinematics)
    # Wir müssen h so berechnen, dass y_global = y_target bleibt.
    # Umkehrung der y_local Formel:
    # y = -(R + Lv) * sin(p) + rollOffset + h * cos(p)
    # => h = (y_target + (R + Lv)*sin(p) - rollOffset) / cos(p)

    res_list_z = []
    res_list_pitch = []

    # Wir nutzen ein h-Limit-Check, da nicht jedes v bei festem y erreichbar ist
    for v in v_vals:
        # Wir berechnen zuerst den Pitch für dieses v (h hat darauf keinen Einfluss)
        # Hier simulieren wir kurz den lokalen Pitch-Wert:
        alpha_off = np.arctan(kin_model.hAxis_vertical / kin_model.hAxis_horizontal)
        pitch_rad = np.arcsin((kin_model.hAxis_vertical + v) / kin_model.hAxis_diagonal) - alpha_off

        # rollOffset berechnen
        roll_off = kin_model.hAxis_horizontal - np.sqrt(
            kin_model.hAxis_diagonal ** 2 - (kin_model.hAxis_vertical + v) ** 2)

        # Erforderliches h berechnen, um y_target zu halten
        cos_p = np.cos(pitch_rad)
        h_req = (y_target + (kin_model.radius + kin_model.hAxis_vertical) * np.sin(pitch_rad) - roll_off) / cos_p

        # Prüfen, ob dieses h mechanisch möglich ist
        if limits['h'][0] <= h_req <= limits['h'][1]:
            # Wenn ja, berechnen wir das resultierende Z
            z_val = -kin_model.radius * (1 - np.cos(pitch_rad)) - (h_req * np.sin(pitch_rad))

            res_list_z.append(z_val)
            res_list_pitch.append(np.degrees(pitch_rad))

    # 3. Plotten
    plt.figure(figsize=(10, 6))
    plt.plot(res_list_pitch, res_list_z, color='red', linewidth=2, label=f'Trajektorie bei Y={y_target}mm')

    plt.title(f'Kopplung: Vertikalbewegung (Z) zu Neigung (Pitch)\nKonstante Längsposition Y = {y_target} mm',
              fontsize=12)
    plt.xlabel('Pitch [Grad] (durch V-Achse)', fontsize=10)
    plt.ylabel('Klinisches Z (Vertical) [mm]', fontsize=10)

    # Markiere den Nullpunkt (Home)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.legend()

    # Textbox mit Infos
    info_text = f"Spannen bei Y={y_target}:\nΔZ: {max(res_list_z) - min(res_list_z):.1f} mm\nΔPitch: {max(res_list_pitch) - min(res_list_pitch):.1f}°"
    plt.gca().text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()


if __name__ == "__main__":
    LIMITS = {
        'h': (-45, 25),
        'v': (-35, 50),
        'r': (-30, 120)
    }

    kin_model = SurfKinematics()

    # Grid Dichte auf 10 reduziert für super performantes Rendering im Browser
    plot_workspace(kin_model, LIMITS, couch_angles=[0.0, 45.0, 90.0], points_per_axis=30)

    Y_CONSTANT = 0.0
    plot_z_pitch_behavior(kin_model, LIMITS, y_target=Y_CONSTANT)