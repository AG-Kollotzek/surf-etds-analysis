import numpy as np


class SurfKinematics:
    def __init__(self):
        # 1. Konstanten & Bauteilmaße (Idealisiert, ohne Unsicherheiten)
        # Hebel der H-Achse
        self.hAxis_horizontal = 4.75 + 9.5 + 138 + 9.5 + 9.2 + 14.5
        self.hAxis_vertical = 14.5 + 20 + 10
        self.hAxis_diagonal = np.sqrt(self.hAxis_horizontal ** 2 + self.hAxis_vertical ** 2)

        # Distanzen Phantom zu H-Schlitten in Home-Position
        rotatationTable_height = 130
        phantomCenterDistance = 75
        phantomToTableDistance = 21
        self.sliderShift = 30 + 16
        self.radius = rotatationTable_height + phantomToTableDistance + phantomCenterDistance + self.sliderShift

    def calculate_task_space(self, h_raw, v_raw, r_raw_deg, couch_angle_raw_deg=0.0):
        """
        Berechnet die globalen ETD-Koordinaten aus den physischen Achsenwerten.
        """
        # Konvertierung zu Numpy Arrays für Vektor-Berechnungen
        h_u = np.asarray(h_raw)
        v_u = np.asarray(v_raw)
        r_u = np.deg2rad(np.asarray(r_raw_deg))

        # ==========================================
        # SCHRITT 1: LOKALE KINEMATIK (Couch = 0°)
        # ==========================================
        alpha_offset = np.arctan(self.hAxis_vertical / self.hAxis_horizontal)
        pitch_rad_local = np.arcsin((self.hAxis_vertical + v_u) / self.hAxis_diagonal) - alpha_offset
        pitch_deg_local = np.degrees(pitch_rad_local)

        rollOffset = self.hAxis_horizontal - np.sqrt(self.hAxis_diagonal ** 2 - (self.hAxis_vertical + v_u) ** 2)

        y_local = -(self.radius + self.hAxis_vertical) * np.sin(pitch_rad_local) + rollOffset + (
                    h_u * np.cos(pitch_rad_local))
        x_local = h_u * 0
        z_local = -self.radius * (1 - np.cos(pitch_rad_local)) - (h_u * np.sin(pitch_rad_local))

        yaw_deg_local = np.degrees(r_u * np.cos(pitch_rad_local))
        roll_deg_local = np.degrees(r_u * np.sin(pitch_rad_local))

        # ==========================================
        # SCHRITT 2: COUCH ROTATION (Transformation)
        # ==========================================
        gamma = np.deg2rad(couch_angle_raw_deg)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)

        x_global = x_local * cos_g - y_local * sin_g
        y_global = x_local * sin_g + y_local * cos_g
        z_global = z_local

        pitch_global = pitch_deg_local * cos_g - roll_deg_local * sin_g
        roll_global = pitch_deg_local * sin_g + roll_deg_local * cos_g
        # BUGFIX: couch_angle in Grad abziehen, nicht gamma (was in rad war!)
        yaw_global = yaw_deg_local - couch_angle_raw_deg

        return {
            'True_Lateral': x_global,
            'True_Longitudinal': y_global,
            'True_Vertical': z_global,
            'True_Pitch': pitch_global,
            'True_Roll': roll_global,
            'True_Yaw': yaw_global
        }