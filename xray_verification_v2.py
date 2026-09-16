"""X-ray (sphere-detection) verification of the phantom kinematics - calibrated model.

Supersedes xray_verification_csv.py. That script and its four output CSVs
(xray_verification_table / _relative / _with_errors / _with_errors_new) are all stale: they were
produced with the UNcalibrated kinematics_werror. Note that xray_verification_csv.py is now
actively broken as well - run against the current model it would pair a sign-compensated
Position = -10 with an uncompensated model and produce a ~20 mm discrepancy. Four things are
fixed here:

  1. Kinematics. Uses kinematics_werror_v2.SurfKinematics, i.e. the empirically calibrated lever
     arm (SD_CALIB_DEFAULT = 4.156 +/- 0.550 mm). The old script imported kinematics_werror.

  2. H-axis sign. The hard-coded Position = -10 in the old script is a leftover compensation from
     an earlier state of the model, not from kinematics_werror as it stands today - that file
     already carries the correct - h*cos(pitch) term (the +h*cos(pitch) version survives only in
     the stale kinematics.py, which nothing in the production chain imports). Against the current
     model the -10 is simply wrong: the raw terminal log of this campaign
     (surf-etds-data: campaigns/2026-03-10_L4/phantom/messung_20260310_172148.csv) shows the axis dwelled at
     Pos_H = +10.00 mm. Commanded values are therefore taken from the raw log rather than
     hard-coded, and verify_commanded_against_log() re-checks them in order on every run.

  3. R-axis sign. DataConverter.load_csv inverts Pos_R in legacy mode; the old script fed the raw
     terminal value straight into the model. Applied here as well, so this script and the dynamic
     pipeline share one convention. It has no numerical effect while V = 0 (the model produces no
     translation from R alone), but it stops being harmless the moment any R-dependent term is
     added, so it is applied rather than left as a latent trap.

  4. Zero reference. Each point is referenced to the nearest PRECEDING zero measurement instead of
     the first row of its block - see build_table.

The measured column stays hand-transcribed: there is NO ETD tracking JSON for this campaign
(the log jumps 16:40:07 -> 17:39:28), so the 17 ExacTrac stereoscopic-kV readouts below come from
the measurement protocol. They are the independent radiographic reference - not surface tracking.

No eccentricity term is modelled. The rotation block shows a systematic in-plane drift growing to
~1.5 mm at 90 deg, consistent with the tracked centroid sitting ~1.1 mm off the rotation axis, but
that offset is specific to one phantom build (axis adjustment, 3D-printed head-to-platform
connector) and is not stable across rebuilds, so it is reported rather than absorbed into the
model.

Output: xray_verification_v2.csv
"""

import glob

import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy as unp

from kinematics_werror_v2 import SurfKinematics

RAW_LOG_GLOB = 'surf-etds-data/campaigns/2026-03-10_L4/phantom/*172148.csv'
OUT_CSV = 'xray_verification_v2.csv'

# 1-sigma of a single ExacTrac coordinate. The readout is quoted to 0.1 mm; taking that as the
# 2-sigma interval gives 1-sigma = 0.05 mm.
ET_SINGLE_SIGMA = 0.05

# Hand-transcribed ExacTrac stereoscopic-kV readouts (lateral, longitudinal, vertical) in mm,
# in the chronological order of the dwell points. Block = the zero reference each point is
# measured against.
MEASURED = [
    ('Longitudinal', [0.2, -0.3, 0.2], 'Longitudinal'),
    ('Longitudinal', [0.3, -10.3, 0.2], 'Longitudinal'),
    ('Longitudinal', [0.2, -0.3, 0.2], 'Longitudinal'),
    ('Vertical', [0.2, -0.3, 0.2], 'Vertical'),
    ('Vertical', [0.7, -14.3, -0.2], 'Vertical'),
    ('Vertical', [-0.4, 14.1, -0.3], 'Vertical'),
    ('Vertical', [0.2, -0.3, 0.2], 'Vertical'),
    ('Rotation', [0.1, -0.3, 0.2], 'Rotation'),
    ('Rotation', [0.2, -0.3, 0.2], 'Rotation'),
    ('Rotation', [0.1, -0.4, 0.2], 'Rotation'),
    ('Rotation', [0.2, -0.2, 0.2], 'Rotation'),
    ('Rotation', [0.2, -0.3, 0.2], 'Rotation'),
    ('Rotation', [-0.2, -0.7, 0.2], 'Rotation'),
    ('Rotation', [0.6, -0.1, 0.2], 'Rotation'),
    ('Rotation', [-0.3, -1.1, 0.2], 'Rotation'),
    ('Rotation', [-0.1, -1.8, 0.2], 'Rotation'),
    ('Rotation', [-0.3, -1.4, 0.2], 'Rotation'),
]

# Commanded axis triples (H, V, R_raw) for the 17 points, read off the raw terminal log's dwell
# plateaus. See module docstring - these replace the hard-coded, sign-compensated literals.
COMMANDED = [
    (0.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 10.0, 0.0),
    (0.0, -10.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 4.95),
    (0.0, 0.0, -4.95),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 29.96),
    (0.0, 0.0, -29.96),
    (0.0, 0.0, 45.0),
    (0.0, 0.0, 90.0),
    (0.0, 0.0, 59.98),
]

# The deflections that the all-axes dynamic sequence actually exercises. Only these are
# comparable with the per-DoF RMSE of meas_01..03.
ALL_AXES_POINTS = (2, 5, 6, 10, 11)   # 1-based Point_ID


def dwell_sequence(df, min_sec=2.0):
    """Chronological list of (H, V, R) triples the terminal actually held for >= min_sec.

    Transit samples are excluded by requiring the triple to be constant for min_sec, so a value
    that the axis merely passed through cannot be mistaken for a commanded dwell.
    """
    t = df['Time_Sec'].values
    A = df[['Pos_H', 'Pos_V', 'Pos_R']].values.round(2)
    out, start = [], 0
    for i in range(1, len(A) + 1):
        if i == len(A) or not np.array_equal(A[i], A[start]):
            if t[min(i, len(t) - 1)] - t[start] >= min_sec:
                out.append(tuple(float(x) for x in A[start]))
            start = i
    return out


def verify_commanded_against_log():
    """Cross-check COMMANDED against the raw terminal log, ORDER INCLUDED.

    Set membership alone is not enough: it would accept a reordered table, a swapped H/V column, or
    a value the axis only passed through in transit. The measured triples are hand-transcribed and
    are matched to the commanded ones positionally, so a misordering would silently pair each
    ExacTrac reading with the wrong axis position. This walks the log's dwell sequence in order and
    requires COMMANDED to be a subsequence of it.
    """
    hits = glob.glob(RAW_LOG_GLOB, recursive=True)
    if not hits:
        print("[!] raw log not found - skipping the cross-check against the terminal data.")
        return False
    df = pd.read_csv(hits[0], sep=';', decimal='.')
    seq = dwell_sequence(df)

    # The protocol has 17 imaging points but the terminal held only 16 distinct dwells: the zero
    # between two blocks was imaged twice (closing one block, opening the next). A consecutive
    # repeat of the same triple therefore maps to the SAME dwell rather than consuming the next
    # one. Any other order violation still fails.
    j, matched, repeats = 0, [], 0
    for k, triple in enumerate(COMMANDED):
        if matched and triple == COMMANDED[k - 1]:
            matched.append(matched[-1])
            repeats += 1
            continue
        while j < len(seq) and seq[j] != triple:
            j += 1
        if j >= len(seq):
            print(f"[!] point {k + 1} {triple} does not appear (in order) in the log's dwell "
                  f"sequence. The COMMANDED table is out of step with {hits[0]}.")
            print(f"    log dwell sequence: {seq}")
            return False
        matched.append(j)
        j += 1

    print(f"[+] all {len(COMMANDED)} commanded triples matched IN ORDER against the "
          f"{len(seq)} dwell plateaus of {hits[0]}"
          + (f" ({repeats} zero position(s) imaged twice)" if repeats else ""))
    return True


def build_table(terminal_version='legacy'):
    kin = SurfKinematics()

    rows = []
    for idx, ((block_axis, meas, block), (h, v, r_raw)) in enumerate(zip(MEASURED, COMMANDED)):
        r = -r_raw if terminal_version == 'legacy' else r_raw   # as in DataConverter.load_csv
        calc = kin.calculate_task_space(np.array([h]), np.array([v]), np.array([r]), 0.0)
        rows.append({
            'ID': idx, 'Axis': block_axis, 'Block': block,
            'H_mm': h, 'V_mm': v, 'R_deg': r_raw,
            'C_Lat': calc['True_Lateral'][0],
            'C_Long': calc['True_Longitudinal'][0],
            'C_Vert': calc['True_Vertical'][0],
            'M_Lat': ufloat(meas[0], ET_SINGLE_SIGMA),
            'M_Long': ufloat(meas[1], ET_SINGLE_SIGMA),
            'M_Vert': ufloat(meas[2], ET_SINGLE_SIGMA),
        })

    # Each point is referenced to the NEAREST PRECEDING zero-position measurement, so the absolute
    # setup offset drops out and only the MOVEMENT vector is compared. Not the first row of the
    # block: the rotation block opens with a zero reading of (0.1, -0.3, 0.2) while every other
    # zero in the protocol reads (0.2, -0.3, 0.2), so anchoring the whole block on it injected a
    # spurious -0.1 mm lateral offset into all seven rotation rows. Using the closest preceding
    # zero also tracks any slow drift between blocks, and matches how the QA repo's
    # sphere_detection.select_deflection_rows picks its reference.
    def _is_zero(r):
        return r['H_mm'] == 0.0 and r['V_mm'] == 0.0 and r['R_deg'] == 0.0

    baseline_for = {}
    last_zero = None
    for row in rows:
        if _is_zero(row):
            last_zero = row
        if last_zero is None:
            raise ValueError(f"Point {row['ID'] + 1} has no preceding zero-position measurement.")
        baseline_for[row['ID']] = last_zero

    table = []
    for row in rows:
        base = baseline_for[row['ID']]
        rel_c = [row['C_Lat'] - base['C_Lat'], row['C_Long'] - base['C_Long'],
                 row['C_Vert'] - base['C_Vert']]
        rel_m = [row['M_Lat'] - base['M_Lat'], row['M_Long'] - base['M_Long'],
                 row['M_Vert'] - base['M_Vert']]
        diff = [c - m for c, m in zip(rel_c, rel_m)]
        mag = float(np.sqrt(sum(d.n ** 2 for d in diff)))

        entry = {'Point_ID': row['ID'] + 1, 'Axis': row['Axis'],
                 'H_mm': row['H_mm'], 'V_mm': row['V_mm'], 'R_deg': row['R_deg']}
        for name, vals in (('Model', rel_c), ('ExacTrac', rel_m), ('Discrepancy', diff)):
            for comp, val in zip('XYZ', vals):
                entry[f'{name}_{comp}_mm'] = round(val.n, 3)
                entry[f'{name}_{comp}_sd'] = round(val.s, 3)
        entry['Discrepancy_3D_mm'] = round(mag, 3)
        entry['In_AllAxes_Range'] = row['ID'] + 1 in ALL_AXES_POINTS
        table.append(entry)

    return pd.DataFrame(table)


def main():
    verify_commanded_against_log()
    df = build_table()
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[>] {OUT_CSV}")

    cols = ['Point_ID', 'Axis', 'H_mm', 'V_mm', 'R_deg',
            'Model_Y_mm', 'ExacTrac_Y_mm',
            'Discrepancy_X_mm', 'Discrepancy_Y_mm', 'Discrepancy_Z_mm', 'Discrepancy_3D_mm']
    print("\nFull table (relative to each block's zero reference):")
    print(df[cols].to_string(index=False))

    sub = df[df['In_AllAxes_Range'] & (df[['H_mm', 'V_mm', 'R_deg']].abs().sum(axis=1) > 0)]
    print("\nPoints inside the all-axes range - the deflections comparable with the per-DoF RMSE of"
          "\nmeas_01..03. Note the X-ray campaign imaged H = +10 mm only (the dynamic runs sweep"
          "\nH both ways), V = +/-10 mm and R = +/-5 deg:")
    print(sub[cols].to_string(index=False))
    print(f"\n  max |3D discrepancy| in that range : {sub['Discrepancy_3D_mm'].max():.3f} mm")
    print(f"  RMS of the 3D discrepancies        : "
          f"{np.sqrt(np.mean(sub['Discrepancy_3D_mm'].values ** 2)):.3f} mm")

    rot = df[(df['Axis'] == 'Rotation') & (df['R_deg'].abs() > 5)]
    if not rot.empty:
        print(f"\n  Rotation points beyond +/-5 deg (outside the dynamic range, unmodelled "
              f"centroid offset):\n  max |3D discrepancy| = {rot['Discrepancy_3D_mm'].max():.3f} mm "
              f"at R = {rot.loc[rot['Discrepancy_3D_mm'].idxmax(), 'R_deg']:.0f} deg")


if __name__ == '__main__':
    main()
