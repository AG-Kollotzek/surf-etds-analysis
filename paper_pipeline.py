"""Re-evaluation of the 2026-03-10 campaign with the calibrated kinematics and the QA methodology.

What changed relative to the previous evaluation (raw_data_export.py + calculate_rmse.py):

  1. Kinematics   kinematics_werror_v2.SurfKinematics - empirically calibrated lever arm
                  (SD_CALIB_DEFAULT = 4.156 +/- 0.550 mm, fitted against the X-ray sphere
                  detection of all four linacs) and the exact rotation composition
                  R_x(pitch) @ R_z(r) instead of the small-angle approximation.
  2. Alignment    sync pulses located by value stability and anchored on their 50 % entry
                  EDGES, then a least-squares refinement of time scale and offset against the
                  full longitudinal curve (PC clock drift), instead of two plateau midpoints.
  3. No baseline  the old pipeline subtracted a median offset per DoF before computing errors,
                  which removed any genuine static offset from the RMSE. Dropped.
  4. Vertical sign the ETD vertical axis is inverted w.r.t. the phantom kinematics. The old
                  calculate_rmse.py compared them unflipped (batch_evaluation.py did flip it for
                  plotting - the two disagreed). Now handled centrally in qa_metrics.DOF_SPEC.
  5. Window       metrics are taken strictly between the sync pulses (+/- 3 s margin) instead of
                  over the whole cropped record.

Outputs (default root paper_data/process_v2, leaving the previous export untouched):

  <root>/<RT|32>/<Group>/meas_<id>/02_after_align_etd.csv
                                  /02_after_align_phantom.csv
                                  /03_alldof_rmse.csv         per-DoF MAE / RMSE / MaxAE
                                  /03_residual3d.csv          signed residuals + 3D magnitudes
                                  /metadata.json              run metadata + sync_info
  <root>/<RT|32>/<Group>/04_group_rmse_summary.csv            RMSE mean +/- SD over the runs
  <root>/05_weighted_rmse_<RT|32>.csv                         combined across H/V/R
  <root>/06_temperature_comparison.csv                        RT vs 32 C, per group and DoF

Usage:
    python paper_pipeline.py                 # everything (RT and 32 C)
    python paper_pipeline.py --pads OFF      # room temperature only
    python paper_pipeline.py --groups Horizontal Vertical
    python paper_pipeline.py --no-refine     # disable the least-squares time refinement
"""

import argparse
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from DataConverter import ETDQAProcessor, MEASUREMENT_10032026
import qa_metrics as qm
import paper_plots as pp

# Raw data come from the surf-etds-data submodule (git submodule update --init).
RAW_DIR = Path('surf-etds-data/campaigns/2026-03-10_L4')
DEFAULT_OUT = Path('paper_data/process_v2')

# The three isolated-axis campaigns that are combined into the single weighted bar chart.
SINGLE_AXIS_GROUPS = ('Horizontal', 'Vertical', 'Rotation')

ETD_EXPORT_COLS = ['Time_Sec', 'lateral', 'longitudinal', 'vertical', 'pitch', 'yaw', 'roll']
PHANTOM_EXPORT_COLS = ['Time_Sec', 'True_Lateral', 'True_Longitudinal', 'True_Vertical',
                       'True_Pitch', 'True_Yaw', 'True_Roll']


def pads_folder(heatingpads):
    """'OFF' -> 'RT', '32' -> '32'."""
    return 'RT' if heatingpads == 'OFF' else str(heatingpads)


def find_raw_paths(meta):
    """Locate the SURF terminal CSV (phantom/) and the ETD tracking JSON (etd/) for one measurement entry."""
    csv_hits = sorted((RAW_DIR / 'phantom').glob(f"*{meta['CSV']}.csv"))
    etd = meta['ETD']
    json_hits = sorted((RAW_DIR / 'etd').glob(f"*{etd[:2]}-{etd[2:4]}-{etd[4:]}.json"))
    if len(csv_hits) != 1 or len(json_hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one raw file each (CSV {meta['CSV']}: {len(csv_hits)}, ETD {etd}: {len(json_hits)}) "
            f"in {RAW_DIR}; run 'git submodule update --init'")
    return str(csv_hits[0]), str(json_hits[0])


def process_run(meas_id, meta, refine=True, margin_sec=qm.DEFAULT_WINDOW_MARGIN_SEC):
    """Full chain for one run: load -> calibrated kinematics -> align -> metrics.

    Returns a dict with the aligned frames, the per-DoF metrics, the residual time series and
    the sync information, or raises.
    """
    csv_path, json_path = find_raw_paths(meta)

    proc = ETDQAProcessor(terminal_version='legacy')
    proc.load_csv(csv_path)
    proc.load_json(json_path)
    proc.apply_kinematics(couch_angle=0.0)
    proc.align_and_crop_signals(pad_sec=5.0, couch_angle=0.0, refine=refine)

    window = qm.measurement_window(proc.sync_info, margin_sec=margin_sec)
    metrics = qm.compute_dof_metrics(proc.df_csv, proc.df_json, window)
    residuals = qm.compute_residual_vectors(proc.df_csv, proc.df_json, window)

    # Tracking-lost frames are absent from the ETD log, so the interpolation onto the phantom grid
    # bridges straight across them: the residual there is invented, not measured. Counted and
    # reported rather than silently dropped. Measured over this campaign, dropouts occur ONLY in
    # the vertical-slide series - no reported group has a single one inside its window.
    lost = getattr(proc, 'lost_times_aligned', [])
    lost_in_window = [t for t in lost if window[0] <= t <= window[1]]

    return {
        'meas_id': meas_id,
        'meta': meta,
        'df_csv': proc.df_csv,
        'df_json': proc.df_json,
        'sync_info': proc.sync_info,
        'window': window,
        'metrics': metrics,
        'residuals': residuals,
        'lost_times': lost,
        'lost_in_window': lost_in_window,
        'csv_path': csv_path,
        'json_path': json_path,
    }


def export_run(run, out_root):
    """Write the per-run artefacts and return the measurement directory."""
    meta = run['meta']
    group = meta['Gruppe'].replace(' ', '_')
    meas_dir = Path(out_root) / pads_folder(meta['Heatingpads']) / group / f"meas_{int(run['meas_id']):02d}"
    meas_dir.mkdir(parents=True, exist_ok=True)

    run['df_json'][[c for c in ETD_EXPORT_COLS if c in run['df_json'].columns]] \
        .to_csv(meas_dir / '02_after_align_etd.csv', index=False, sep=';', decimal='.')
    run['df_csv'][[c for c in PHANTOM_EXPORT_COLS if c in run['df_csv'].columns]] \
        .to_csv(meas_dir / '02_after_align_phantom.csv', index=False, sep=';', decimal='.')

    run['metrics'].to_csv(meas_dir / '03_alldof_rmse.csv', index=False, sep=';', decimal='.')
    run['residuals'].to_csv(meas_dir / '03_residual3d.csv', index=False, sep=';', decimal='.')

    payload = dict(meta)
    payload.update({
        'meas_id': run['meas_id'],
        'sync_info': run['sync_info'],
        'window_start_sec': run['window'][0],
        'window_end_sec': run['window'][1],
        'lost_times_aligned': list(run['lost_times']),
        'tracking_lost_in_window': len(run.get('lost_in_window', [])),
        'csv_file': Path(run['csv_path']).name,
        'json_file': Path(run['json_path']).name,
        'kinematics': 'kinematics_werror_v2.SurfKinematics (calibrated)',
        # The ETD export keeps the raw logged columns. 'vertical' is therefore in the ETD's own
        # sign convention, which is INVERTED w.r.t. the phantom kinematics: anyone recomputing a
        # residual from 02_after_align_etd.csv must negate it first (qa_metrics.DOF_SPEC does this
        # via etd_sign). Without the flip the vertical RMSE comes out several times too large.
        'etd_sign_convention': {d.name: d.etd_sign for d in qm.DOF_SPEC},
    })
    with open(meas_dir / 'metadata.json', 'w') as f:
        json.dump(payload, f, indent=2)

    return meas_dir


def select_measurements(groups=None, pads=None):
    """{(group, heatingpads): [meas_id, ...]} filtered by the requested groups / pad states."""
    out = {}
    for mid, meta in MEASUREMENT_10032026.items():
        if groups and meta['Gruppe'] not in groups:
            continue
        if pads and meta['Heatingpads'] not in pads:
            continue
        out.setdefault((meta['Gruppe'], meta['Heatingpads']), []).append(mid)
    for key in out:
        out[key].sort(key=int)
    return out


def run_campaign(groups=None, pads=None, refine=True, out_root=DEFAULT_OUT, verbose=True):
    """Process every selected run, export it, and build the group summaries.

    Returns {(group, heatingpads): {'runs': [...], 'summary': DataFrame}}.
    """
    out_root = Path(out_root)
    selection = select_measurements(groups, pads)
    results = {}

    for (group, pad), ids in sorted(selection.items()):
        runs, failures = [], []
        for mid in ids:
            meta = MEASUREMENT_10032026[mid]
            try:
                run = process_run(mid, meta, refine=refine)
            except Exception as exc:
                failures.append((mid, str(exc)))
                if verbose:
                    print(f"   [X] meas_{mid} ({group}, pads={pad}): {exc}")
                continue
            export_run(run, out_root)
            runs.append(run)

        if not runs:
            if verbose:
                print(f"[!] {group} / pads={pad}: no usable run, skipped.")
            continue

        summary = qm.aggregate_runs([r['metrics'] for r in runs])
        group_dir = out_root / pads_folder(pad) / group.replace(' ', '_')
        group_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(group_dir / '04_group_rmse_summary.csv', index=False, sep=';', decimal='.')

        results[(group, pad)] = {'runs': runs, 'summary': summary, 'failures': failures}
        if verbose:
            n = len(runs)
            print(f"[OK] {group:26s} pads={pad:3s}  {n} run(s)"
                  + (f"  ({len(failures)} failed)" if failures else ""))
            n_lost = sum(len(r.get('lost_in_window', [])) for r in runs)
            if n_lost:
                print(f"     [!] {n_lost} tracking-lost frame(s) inside the evaluation window across "
                      f"{sum(1 for r in runs if r.get('lost_in_window'))} run(s). The ETD has no data "
                      f"there, so the residual is interpolated across the dropout rather than "
                      f"measured.")
            flagged = summary[summary.get('Reference_Offset_Exceeded', False) == True] \
                if 'Reference_Offset_Exceeded' in summary.columns else summary.iloc[0:0]
            if not flagged.empty:
                worst = flagged.loc[flagged['Reference_Offset_max'].idxmax()]
                print(f"     [!] commanded/ETD origin mismatch up to "
                      f"{worst['Reference_Offset_max']:.1f} {worst['Unit']} "
                      f"({', '.join(flagged['DoF'])}). The ETD zeroes its shifts on its own "
                      f"reference capture; this run does not start at the kinematic home, so these "
                      f"RMSE values are a STATIC OFFSET, not a tracking accuracy. Excluded from the "
                      f"combined figures - do not quote them as accuracy.")

    return results


def weighted_across_axes(results, pad, groups=SINGLE_AXIS_GROUPS):
    """Combined per-DoF RMSE across the isolated-axis campaigns, both weightings."""
    by_group = {}
    for (group, p), payload in results.items():
        if p != pad or group not in groups:
            continue
        by_group[group] = [r['metrics'] for r in payload['runs']]
    if not by_group:
        return pd.DataFrame()

    pooled = qm.combine_groups(by_group, weighting='pooled')
    invvar = qm.combine_groups(by_group, weighting='inverse_variance')
    return pd.concat([pooled, invvar], ignore_index=True)


def temperature_comparison(results, groups=SINGLE_AXIS_GROUPS):
    """RT vs 32 C per group and DoF, from the group summaries."""
    rows = []
    for group in groups:
        rt = results.get((group, 'OFF'))
        hot = results.get((group, '32'))
        if rt is None or hot is None:
            continue
        s_rt = rt['summary'].set_index('DoF')
        s_hot = hot['summary'].set_index('DoF')
        for dof in s_rt.index:
            if dof not in s_hot.index:
                continue
            a = float(s_rt.loc[dof, 'RMSE_mean'])
            b = float(s_hot.loc[dof, 'RMSE_mean'])
            rows.append({
                'Group': group,
                'DoF': dof,
                'Unit': s_rt.loc[dof, 'Unit'],
                'RMSE_RT': a,
                'SD_RT': float(s_rt.loc[dof, 'RMSE_std']),
                'RMSE_32C': b,
                'SD_32C': float(s_hot.loc[dof, 'RMSE_std']),
                'Delta_32C_minus_RT': b - a,
                'Relative_change_pct': 100.0 * (b - a) / a if a > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--groups', nargs='*', default=None,
                    help="restrict to these groups (default: all)")
    ap.add_argument('--pads', nargs='*', default=None, choices=['OFF', '32'],
                    help="restrict to these heating-pad states (default: both)")
    ap.add_argument('--out', default=str(DEFAULT_OUT), help="output root")
    ap.add_argument('--no-refine', action='store_true',
                    help="disable the least-squares time-alignment refinement")
    args = ap.parse_args()

    out_root = Path(args.out)
    print(f"=== Re-evaluation with calibrated kinematics -> {out_root} ===\n")

    results = run_campaign(groups=args.groups, pads=args.pads,
                           refine=not args.no_refine, out_root=out_root)

    if not results:
        print("\nNothing processed.")
        return

    for pad in ('OFF', '32'):
        table = weighted_across_axes(results, pad)
        if table.empty:
            continue
        path = out_root / f"05_weighted_rmse_{pads_folder(pad)}.csv"
        table.to_csv(path, index=False, sep=';', decimal='.')
        print(f"\n[>] combined across {', '.join(SINGLE_AXIS_GROUPS)} (pads={pad}) -> {path}")
        pooled = table[table['Weighting'] == 'pooled']
        for _, r in pooled.iterrows():
            print(f"      {r['DoF']:14s} {r['RMSE_combined']:.3f} {r['Unit']}"
                  f"   between-axis SD {r['SD_between_runs']:.3f}"
                  f" | within-axis SD {r['SD_within_axis']:.3f}"
                  f"   (n = {r['N_Samples_total']})")

    temps = temperature_comparison(results)
    if not temps.empty:
        path = out_root / '06_temperature_comparison.csv'
        temps.to_csv(path, index=False, sep=';', decimal='.')
        print(f"\n[>] RT vs 32 C -> {path}")

    make_figures(results, out_root)
    print("\nDone.")


FIG_DIR_NAME = 'figures'
ZOOM_CONFIG = Path('zoom_box_config_v2.json')
PLOT_BIN_SEC = 0.2   # same 200 ms binning the previous batch_evaluation used


def combine_runs_for_plot(runs, bin_sec=PLOT_BIN_SEC):
    """Put the repeated runs of one group on a common time base and average them.

    Time is measured relative to the centre of each run's FIRST sync pulse, which is the one
    event both systems observe and which the alignment anchors on - so the runs overlay
    correctly even though each has its own absolute clock.

    Panels 1+2 get the mean +/- SD of the signed curves. The 3D residual is handled differently
    on purpose: the magnitude is formed per run FIRST and only then averaged, so that residuals
    of opposite sign in different runs cannot cancel.
    """
    if not runs:
        raise ValueError("no runs to combine")

    def rel(run, t):
        return np.asarray(t, dtype=float) - float(run['sync_info']['aligned_first_mid'])

    # common grid = overlap of all runs
    lo = max(rel(r, r['df_csv']['Time_Sec'].values[0]) for r in runs)
    hi = min(rel(r, r['df_csv']['Time_Sec'].values[-1]) for r in runs)
    lo = max(lo, max(rel(r, r['df_json']['Time_Sec'].values[0]) for r in runs))
    hi = min(hi, min(rel(r, r['df_json']['Time_Sec'].values[-1]) for r in runs))
    if not hi > lo:
        raise ValueError("runs do not overlap on the common time base")
    t = np.arange(lo, hi, bin_sec)

    etd_stack, cmd_stack = {}, {}
    for dof in qm.DOF_SPEC:
        etd_stack[dof.name] = np.vstack([
            np.interp(t, rel(r, r['df_json']['Time_Sec'].values), dof.etd_values(r['df_json']))
            for r in runs])
        cmd_stack[dof.name] = np.vstack([
            np.interp(t, rel(r, r['df_csv']['Time_Sec'].values), r['df_csv'][dof.phantom_col].values)
            for r in runs])

    mean = {'etd': {k: v.mean(axis=0) for k, v in etd_stack.items()},
            'commanded': {k: v.mean(axis=0) for k, v in cmd_stack.items()}}
    std = {'etd': {k: v.std(axis=0, ddof=1) if v.shape[0] > 1 else np.zeros_like(v[0])
                   for k, v in etd_stack.items()},
           'commanded': {k: v.std(axis=0, ddof=1) if v.shape[0] > 1 else np.zeros_like(v[0])
                         for k, v in cmd_stack.items()}}

    etd_rms = {}
    for col, key in (('rmse3d', 'rmse3d'), ('rmse_temp', 'rmse_temp')):
        if col in runs[0]['df_json'].columns:
            etd_rms[key] = np.vstack([
                np.interp(t, rel(r, r['df_json']['Time_Sec'].values),
                          r['df_json'][col].values) for r in runs]).mean(axis=0)
        else:
            etd_rms[key] = np.zeros_like(t)

    # 3D residual: magnitude per run, then mean +/- SD, restricted to the evaluation window
    r_lo = max(rel(r, r['residuals']['Time_Sec'].values[0]) for r in runs)
    r_hi = min(rel(r, r['residuals']['Time_Sec'].values[-1]) for r in runs)
    t_res = np.arange(r_lo, r_hi, bin_sec)
    trans = np.vstack([np.interp(t_res, rel(r, r['residuals']['Time_Sec'].values),
                                 r['residuals']['Resid3D_Trans'].values) for r in runs])
    rot = np.vstack([np.interp(t_res, rel(r, r['residuals']['Time_Sec'].values),
                               r['residuals']['Resid3D_Rot'].values) for r in runs])
    n = len(runs)
    residual = {
        't': t_res,
        'trans_mean': trans.mean(axis=0),
        'trans_std': trans.std(axis=0, ddof=1) if n > 1 else np.zeros_like(t_res),
        'rot_mean': rot.mean(axis=0),
        'rot_std': rot.std(axis=0, ddof=1) if n > 1 else np.zeros_like(t_res),
    }

    w0 = float(np.mean([r['window'][0] - r['sync_info']['aligned_first_mid'] for r in runs]))
    w1 = float(np.mean([r['window'][1] - r['sync_info']['aligned_first_mid'] for r in runs]))

    return {'t': t, 'mean': mean, 'std': std, 'etd_rms': etd_rms, 'residual': residual,
            'n_runs': n, 'window': (w0, w1),
            'lost_times': [lt - float(r['sync_info']['aligned_first_mid'])
                           for r in runs for lt in r['lost_times']]}


def load_zoom_config():
    if ZOOM_CONFIG.exists():
        try:
            with open(ZOOM_CONFIG) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_zoom_config(cfg):
    with open(ZOOM_CONFIG, 'w') as f:
        json.dump(cfg, f, indent=2)


# Minimum half-range of a zoom inset, in mm and in degrees. 1 mm / 1 deg is the
# ESTRO-ACROP Table 4 D2 acceptance tolerance for tracking performance, so every inset is
# read against that scale. Without this floor a 0.3 mm deviation blown up to a +/-0.32 mm
# axis looks alarming when it is in fact a third of tolerance.
ZOOM_MIN_HALF_RANGE = 1.0


def auto_zoom_boxes(bundle, width_sec=6.0, margin=1.6):
    """Zoom windows placed where the ETD actually DEVIATES from the commanded motion.

    A window centred on a large excursion is useless: rescaled, it looks exactly like the main
    panel. The two things worth magnifying are both invisible at full scale:

      'coupling'  the secondary translation DoF. While one axis drives +/-10..15 mm, the other two
                  show a sub-millimetre cross-coupling. Placed where that coupling is largest, with
                  y-limits tight around zero so a ~0.5 mm signal fills the inset.
      'plateau'   a commanded HOLD of the dominant DoF, chosen as the plateau with the largest mean
                  |ETD - commanded|. y-limits sit tight around the plateau level, so the steady
                  offset between the dashed and solid trace becomes readable.

    Both get EXPLICIT y-limits - leaving them automatic would rescale to the traces in the window
    and reproduce the main panel again. The half-range never goes below ZOOM_MIN_HALF_RANGE
    (1 mm / 1 deg, the acceptance tolerance), so a sub-millimetre deviation is always seen against
    the tolerance it has to meet rather than being magnified out of proportion. Returns only the boxes it can actually justify; a group
    with no measurable coupling or no plateau simply gets fewer.
    """
    t = bundle['t']
    cmd, etd = bundle['mean']['commanded'], bundle['mean']['etd']
    w0, w1 = bundle.get('window', (t[0], t[-1]))
    inside = (t >= w0) & (t <= w1)
    if inside.sum() < 10:
        inside = np.ones_like(t, dtype=bool)

    trans = ('lateral', 'longitudinal', 'vertical')
    rots = ('pitch', 'yaw', 'roll')
    half = width_sec / 2.0
    boxes = {}

    def _window_around(t_c):
        return [max(float(t[0]), t_c - half), min(float(t[-1]), t_c + half)]

    # --- 'coupling': the two translation DoF that are NOT the driven one ----------------------
    dom_t = max(trans, key=lambda k: float(np.ptp(cmd[k][inside])))
    minor = [k for k in trans if k != dom_t]
    minor_mag = sum(np.abs(etd[k]) for k in minor)
    amp = float(np.max(minor_mag[inside]))
    if amp > 0.15:                      # below ~1.5 quantisation steps there is nothing to show
        t_c = float(t[inside][np.argmax(minor_mag[inside])])
        win = _window_around(t_c)
        sel = (t >= win[0]) & (t <= win[1])
        lim = max(margin * float(np.max(np.abs(np.concatenate(
            [etd[k][sel] for k in minor] + [cmd[k][sel] for k in minor])))),
            ZOOM_MIN_HALF_RANGE)
        boxes['coupling'] = {'ax_idx': 0, 'xlim': win, 'ylim': [-lim, lim],
                             'pos': [0.04, 0.60, 0.28, 0.36], 'active': True}

    # --- 'plateau': a commanded hold, magnified around its own level --------------------------
    # Searched across EVERY DoF of the chosen panel, not just the dominant one: in the all-axes
    # sequence the yaw holds last only ~1.3 s while the pitch holds are longer and show a clearer
    # offset. The plateau with the largest mean |ETD - commanded| wins.
    dom_r = max(rots, key=lambda k: float(np.ptp(cmd[k][inside])))
    if float(np.ptp(cmd[dom_r][inside])) > 1.0:
        ax_idx, candidates, unit_floor = 1, rots, 0.08
    else:
        ax_idx, candidates, unit_floor = 0, trans, 0.25

    avoid = boxes['coupling']['xlim'] if (ax_idx == 0 and 'coupling' in boxes) else None
    best = None
    for key in candidates:
        c = cmd[key]
        amp_k = float(np.max(np.abs(c[inside]))) if inside.any() else 0.0
        if amp_k < 0.5 * unit_floor:
            continue
        moving = np.abs(np.gradient(c, t))
        held = ((moving < 0.05 * max(float(np.max(moving)), 1e-9)) & inside
                & (np.abs(c) > 0.3 * amp_k))
        if not held.any():
            continue
        edges = np.diff(held.astype(int), prepend=0, append=0)
        for i0, i1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
            i1 = min(i1, len(t) - 1)
            if t[i1] - t[i0] < 1.0:
                continue
            t_c = float((t[i0] + t[i1]) / 2.0)
            # Do not stack a second inset on the stretch the coupling box already magnifies.
            if avoid is not None and avoid[0] - half < t_c < avoid[1] + half:
                continue
            resid = float(np.mean(np.abs(etd[key][i0:i1 + 1] - c[i0:i1 + 1])))
            if best is None or resid > best[0]:
                best = (resid, key, i0, i1, t_c)

    if best is not None:
        resid, key, i0, i1, t_c = best
        level = float(np.median(cmd[key][i0:i1 + 1]))
        span = max(4.0 * resid, unit_floor, ZOOM_MIN_HALF_RANGE)
        boxes['plateau'] = {'ax_idx': ax_idx, 'xlim': _window_around(t_c),
                            'ylim': [level - span, level + span],
                            'pos': [0.36 if (ax_idx == 0 and 'coupling' in boxes) else 0.04,
                                    0.60, 0.28, 0.36],
                            'active': True}

    return boxes


def export_three_panel_series(results, out_root):
    """Two full series of three-panel figures: with zoom insets and without.

    Each series is produced in both panel-3 modes, so four PDFs per group:
        <group>_<pads>_etdrms_zoom.pdf     <group>_<pads>_etdrms_nozoom.pdf
        <group>_<pads>_residual3d_zoom.pdf <group>_<pads>_residual3d_nozoom.pdf
    """
    zoom_cfg = load_zoom_config()
    config_dirty = [False]
    made = []

    for (group, pad), payload in sorted(results.items()):
        try:
            bundle = combine_runs_for_plot(payload['runs'])
        except Exception as exc:
            print(f"   [!] {group} / pads={pad}: cannot combine runs ({exc})")
            continue

        pads_label = 'RT' if pad == 'OFF' else f'{pad} C'
        title = (f"ETD surface tracking vs. commanded phantom motion\n"
                 f"{group} | {pads_label} | mean of {bundle['n_runs']} runs")

        # Boxes are keyed per group AND pad state - the old config keyed on the group alone,
        # so the RT and 32 C runs of a group could not have different insets.
        key = f"{group}|{pads_folder(pad)}"
        boxes = zoom_cfg.get(key)
        if boxes is None:
            boxes = auto_zoom_boxes(bundle)
            zoom_cfg[key] = boxes
            config_dirty[0] = True
            print(f"   [i] {key}: no saved zoom boxes - auto-detected defaults written to "
                  f"{ZOOM_CONFIG} (tune them there and re-run).")

        for mode in pp.PANEL3_MODES:
            for with_zoom in (True, False):
                if with_zoom and not boxes:
                    continue   # nothing configured for this group - the no-zoom file covers it
                suffix = 'zoom' if with_zoom else 'nozoom'
                name = f"{group.replace(' ', '_')}_{pads_folder(pad)}_{mode}_{suffix}.pdf"
                sub = Path(out_root) / FIG_DIR_NAME / ('with_zoom' if with_zoom else 'no_zoom')
                made.append(pp.plot_three_panel(
                    bundle, title, sub / name, panel3=mode,
                    zoom_boxes=boxes if with_zoom else None))

    if config_dirty[0]:
        save_zoom_config(zoom_cfg)

    if made:
        print(f"\n[>] {len(made)} three-panel figure(s)")
        for m in made:
            print(f"      {Path(m).parent.name}/{Path(m).name}")
    return made


def make_figures(results, out_root):
    """Bar charts for the talk. Bars use room temperature; 32 C goes into the comparison figure."""
    fig_dir = Path(out_root) / FIG_DIR_NAME
    made = []

    made += export_three_panel_series(results, out_root)

    # Phantom validation: the all-axes sequence, three repeated runs.
    allaxes = results.get(('All axes', 'OFF'))
    if allaxes is not None:
        made.append(pp.plot_rmse_bars(
            allaxes['summary'],
            'All-axes sequence - RMSE per degree of freedom',
            fig_dir / 'fig_allaxes_rmse_RT.pdf',
            subtitle='room temperature | H, V sequential $\\pm$10 mm, R $\\pm$5$\\degree$',
            value_fmt='{:.3f}'))

    # Scanner characterisation: one chart per isolated-axis campaign.
    for group in SINGLE_AXIS_GROUPS:
        payload = results.get((group, 'OFF'))
        if payload is None:
            continue
        made.append(pp.plot_rmse_bars(
            payload['summary'],
            f'{group} axis - RMSE per degree of freedom',
            fig_dir / f'fig_{group.lower()}_rmse_RT.pdf',
            subtitle='room temperature | 5 cycles per run',
            value_fmt='{:.3f}'))

    # The single collapsed chart across all three axis campaigns.
    combined = weighted_across_axes(results, 'OFF')
    if not combined.empty:
        pooled = combined[combined['Weighting'] == 'pooled']
        n_samples = int(pooled['N_Samples_total'].iloc[0])
        n_runs = int(pooled['N_Runs'].iloc[0])
        # Provenance is read back from the table so the caption cannot drift from the data - a
        # group that failed to process must not still be named in the subtitle.
        contributing = str(pooled['Groups'].iloc[0])
        made.append(pp.plot_combined_rmse_bars(
            pooled,
            'Isolated-axis campaigns combined - RMSE per degree of freedom',
            fig_dir / 'fig_combined_rmse_RT.pdf',
            subtitle=(f'sample-weighted over {n_runs} runs '
                      f'({contributing}), N = {n_samples} | room temperature')))

    temps = temperature_comparison(results)
    if not temps.empty:
        made.append(pp.plot_temperature_comparison(
            temps, 'Thermal surface signature: RT vs 32 $\\degree$C', 
            fig_dir / 'fig_temperature_comparison.pdf'))

    if made:
        print(f"\n[>] {len(made)} figure(s) -> {fig_dir}")
        for m in made:
            print(f"      {Path(m).name}")
    return made


if __name__ == '__main__':
    main()
