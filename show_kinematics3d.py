import numpy as np
import plotly.graph_objects as go
from uncertainties import unumpy as unp

# Importiere deine ECHTE Kinematik
from Kinematics import SurfKinematics


def calculate_pure_translation_limits(kin_model, limits, tolerance=0.25):
    """
    Sucht im H/V Aktionsraum nach rein isolierten Translationen.
    Toleranz (mm) bestimmt, wie nah die anderen Achsen an 0 bleiben müssen.
    """
    # Hochauflösendes 2D-Gitter nur für H und V (Rotation beeinflusst XYZ nicht)
    h_dense = np.linspace(limits['h'][0], limits['h'][1], 1500)
    v_dense = np.linspace(limits['v'][0], limits['v'][1], 1500)
    H, V = np.meshgrid(h_dense, v_dense)

    H_flat = H.flatten()
    V_flat = V.flatten()
    R_flat = np.zeros_like(H_flat)  # R auf Null für reine Translation

    # Evaluieren bei Couch = 0
    res = kin_model.calculate_task_space(H_flat, V_flat, R_flat, couch_angle_raw_deg=0.0)

    # Extrahiere Nominalwerte
    y_vals = unp.nominal_values(res['True_Longitudinal'])
    z_vals = unp.nominal_values(res['True_Vertical'])

    pure_limits = {}

    # --- 1. Reine Z-Translation (Y muss 0 sein) ---
    mask_y_zero = np.abs(y_vals) < tolerance
    if np.any(mask_y_zero):
        pure_limits['Z_min'] = np.min(z_vals[mask_y_zero])
        pure_limits['Z_max'] = np.max(z_vals[mask_y_zero])
    else:
        pure_limits['Z_min'], pure_limits['Z_max'] = 0, 0

    # --- 2. Reine Y-Translation (Z muss 0 sein) ---
    mask_z_zero = np.abs(z_vals) < tolerance
    if np.any(mask_z_zero):
        pure_limits['Y_min'] = np.min(y_vals[mask_z_zero])
        pure_limits['Y_max'] = np.max(y_vals[mask_z_zero])
    else:
        pure_limits['Y_min'], pure_limits['Y_max'] = 0, 0

    # --- 3. Reine X-Translation ---
    # Bei Couch = 90 wird das lokale Y zum globalen X.
    # Daher ist die mechanische X-Spanne bei Couch=90 identisch mit der Y-Spanne bei Couch=0.
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


def plot_workspace(kin_model, limits, couch_angles=[0.0, 45.0, 90.0], points_per_axis=12):
    # 1. Pure limits berechnen!
    pure_lims = calculate_pure_translation_limits(kin_model, limits)

    # 2. Reguläres 3D-Raster für die Wolken erstellen
    h_vals = np.linspace(limits['h'][0], limits['h'][1], points_per_axis)
    v_vals = np.linspace(limits['v'][0], limits['v'][1], points_per_axis)
    r_vals = np.linspace(limits['r'][0], limits['r'][1], points_per_axis)

    H, V, R = np.meshgrid(h_vals, v_vals, r_vals)
    H_flat, V_flat, R_flat = H.flatten(), V.flatten(), R.flatten()

    fig_trans = go.Figure()
    fig_rot = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    all_p, all_r, all_yaw = [], [], []

    # 3. Schleife über Couch-Winkel
    for i, angle in enumerate(couch_angles):
        res = kin_model.calculate_task_space(H_flat, V_flat, R_flat, couch_angle_raw_deg=angle)

        x = unp.nominal_values(res['True_Lateral'])
        y = unp.nominal_values(res['True_Longitudinal'])
        z = unp.nominal_values(res['True_Vertical'])
        pitch = unp.nominal_values(res['True_Pitch'])
        roll = unp.nominal_values(res['True_Roll'])
        yaw = unp.nominal_values(res['True_Yaw'])

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

    # ==========================================
    # 4. FADENKREUZE FÜR ISOLIERTE SPANNEN ZEICHNEN
    # ==========================================
    def add_span_line(fig, x0, x1, y0, y1, z0, z1, name, span_val, unit):
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines+text',
            line=dict(color='black', width=5, dash='solid'),  # Jetzt solid und dicker, da es wichtige Werte sind
            text=["", f"Isoliertes Δ {name}: {span_val:.1f} {unit}"],
            textposition="top center",
            textfont=dict(size=13, color='black'),
            name=f"Reine Spanne {name}",
            hoverinfo='skip'
        ))

    # Die Kreuze durch den absoluten Mittelpunkt des Koordinatensystems (0,0,0) legen,
    # da wir nun reine Verschiebungen aus der Nulllage darstellen!
    cx, cy, cz = 0, 0, 0

    add_span_line(fig_trans, pure_lims['X_min'], pure_lims['X_max'], cy, cy, cz, cz, "X",
                  pure_lims['X_max'] - pure_lims['X_min'], "mm")
    add_span_line(fig_trans, cx, cx, pure_lims['Y_min'], pure_lims['Y_max'], cz, cz, "Y",
                  pure_lims['Y_max'] - pure_lims['Y_min'], "mm")
    add_span_line(fig_trans, cx, cx, cy, cy, pure_lims['Z_min'], pure_lims['Z_max'], "Z",
                  pure_lims['Z_max'] - pure_lims['Z_min'], "mm")

    # Rotationen (Bounding Box)
    min_p, max_p = np.min(all_p), np.max(all_p)
    min_r, max_r = np.min(all_r), np.max(all_r)
    min_yaw, max_yaw = np.min(all_yaw), np.max(all_yaw)

    cp, cr, cyaw = 0, 0, 0  # Auch hier Kreuze durch 0,0,0

    add_span_line(fig_rot, min_p, max_p, cr, cr, cyaw, cyaw, "Pitch", max_p - min_p, "°")
    add_span_line(fig_rot, cp, cp, min_r, max_r, cyaw, cyaw, "Roll", max_r - min_r, "°")
    add_span_line(fig_rot, cp, cp, cr, cr, min_yaw, max_yaw, "Yaw", max_yaw - min_yaw, "°")

    # ==========================================
    # 5. LAYOUT ANPASSEN & ANZEIGEN
    # ==========================================
    fig_trans.update_layout(
        title="ETD Phantom: Translations-Aktionsraum (Isolierte Wege durch Nullpunkt markiert)",
        scene=dict(xaxis_title='Lateral (X) [mm]', yaxis_title='Longitudinal (Y) [mm]',
                   zaxis_title='Vertical (Z) [mm]'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig_rot.update_layout(
        title="ETD Phantom: Rotations-Aktionsraum (Absolute Maxima)",
        scene=dict(xaxis_title='Pitch [°]', yaxis_title='Roll [°]', zaxis_title='Yaw [°]'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig_trans.show()
    fig_rot.show()


if __name__ == "__main__":
    LIMITS = {
        'h': (-45, 25),
        'v': (-35, 50),
        'r': (-30, 120)
    }

    kin_model = SurfKinematics()
    plot_workspace(kin_model, LIMITS, couch_angles=[0.0, 45.0, 90.0], points_per_axis=12)