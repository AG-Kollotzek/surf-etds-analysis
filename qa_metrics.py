"""Numerical evaluation of ETD tracking (measured) against SURF phantom kinematics (commanded).

Ported from surf-etds-qa (qa_metrics.py) - that repo carries the more recent, corrected
methodology. Deliberately NOT ported: the QA-report machinery (pass/watch/act classification,
tolerance tables, per-linac report tables, PDF layout). This repository stays a paper-figure and
statistics pipeline; the tolerance verdicts belong to the QA repo.

Added here on top of the ported core:
  * compute_residual_vectors  - 3D residual magnitude over time (translation and rotation triples)
  * aggregate_runs            - per-DoF RMSE mean +/- SD across the repeated runs of one group
  * combine_groups            - single weighted RMSE per DoF across several axis campaigns

Evaluation chain:

  raw CSV + raw JSON
        |  DataConverter.align_and_crop_signals(refine=True)  -> common time base + sync stamps
        v
  aligned frames (no baseline zeroing)
        |  DataConverter.apply_kinematics                     -> True_* clinical columns
        v
        |  compute_dof_metrics / compute_residual_vectors     -> inside the measurement window only
        v
  per-run RMSE  ->  aggregate_runs (mean +/- SD over runs)  ->  combine_groups (weighted)
"""

import numpy as np
import pandas as pd

# Lead-in after the first sync pulse / lead-out before the last that is excluded from the metrics,
# so the sync movement itself and its settling do not enter the error statistics.
DEFAULT_WINDOW_MARGIN_SEC = 3.0


class DofSpec:
    """Pairing of one degree of freedom between the phantom and the ETD data set.

    etd_sign is the critical part: the ETD vertical axis points opposite to the vertical axis of
    the phantom kinematics (and to the sphere-detection Z axis). Without this correction
    'vertical' shows an apparent error of the order of the deflection itself instead of the
    actual residual.
    """

    __slots__ = ('name', 'label', 'phantom_col', 'etd_col', 'etd_sign', 'unit')

    def __init__(self, name, label, phantom_col, etd_col, etd_sign, unit):
        self.name = name
        self.label = label
        self.phantom_col = phantom_col
        self.etd_col = etd_col
        self.etd_sign = etd_sign
        self.unit = unit

    def etd_values(self, df_etd):
        """Raw ETD values brought into the sign convention of the phantom kinematics."""
        return self.etd_sign * df_etd[self.etd_col].values


# Single source of truth for the DoF mapping. Order = plot order (top to bottom).
DOF_SPEC = [
    DofSpec('longitudinal', 'Longitudinal (Y)', 'True_Longitudinal', 'longitudinal', +1.0, 'mm'),
    DofSpec('lateral', 'Lateral (X)', 'True_Lateral', 'lateral', +1.0, 'mm'),
    DofSpec('vertical', 'Vertical (Z)', 'True_Vertical', 'vertical', -1.0, 'mm'),
    DofSpec('pitch', 'Pitch', 'True_Pitch', 'pitch', +1.0, 'deg'),
    DofSpec('yaw', 'Yaw', 'True_Yaw', 'yaw', +1.0, 'deg'),
    DofSpec('roll', 'Roll', 'True_Roll', 'roll', +1.0, 'deg'),
]

DOF_BY_NAME = {d.name: d for d in DOF_SPEC}

# Order of the translational / rotational DoF for the vector triples.
TRANSLATION_DOFS = ('lateral', 'longitudinal', 'vertical')
ROTATION_DOFS = ('pitch', 'yaw', 'roll')


def measurement_window(sync_info, margin_sec=DEFAULT_WINDOW_MARGIN_SEC):
    """Time window of the actual measurement, in the common (aligned) time base.

    Starts margin_sec after the centre of the first sync pulse and ends margin_sec before the
    centre of the last one.

    sync_info: dict as produced by DataConverter.align_and_crop_signals (or read back from
    metadata.json).
    """
    if not sync_info:
        raise ValueError("No sync information available - run align_and_crop_signals() first.")

    t_start = float(sync_info['aligned_first_mid']) + margin_sec
    t_end = float(sync_info['aligned_last_mid']) - margin_sec

    if not t_end > t_start:
        raise ValueError(
            f"Empty measurement window: sync pulses are only "
            f"{t_end - t_start + 2 * margin_sec:.1f}s apart, too little for margin_sec={margin_sec}.")

    return t_start, t_end


def _eval_grid(df_phantom, df_etd, window, dofs):
    """Common evaluation grid and the per-DoF signed residuals on it.

    The phantom (commanded) sample times inside the window are the reference grid; the ETD signal
    is linearly interpolated onto them. Never the other way round, so the commanded trajectory is
    never smoothed by resampling.

    Note on rates: the phantom log is steady at 9.2-9.4 Hz, but the ETD rate varies with the scan
    (7.5-14.5 Hz over this campaign) and is actually SLOWER than the phantom in 15 of the 32 runs.
    In those runs the interpolation up-samples the ETD, so consecutive residuals are correlated and
    N_Samples overstates the number of independent measurements. That matters for any confidence
    interval derived from N, not for the RMSE itself.

    Returns (t_eval, {dof_name: diff}) with diff = measured(ETD) - commanded(phantom).
    """
    t_start, t_end = window
    t_ph = df_phantom['Time_Sec'].values
    mask = (t_ph >= t_start) & (t_ph <= t_end)
    if not mask.any():
        raise ValueError(f"No phantom samples inside the window [{t_start:.2f}, {t_end:.2f}]s.")

    t_eval = t_ph[mask]
    t_etd = df_etd['Time_Sec'].values

    diffs = {}
    for dof in dofs:
        if dof.phantom_col not in df_phantom.columns or dof.etd_col not in df_etd.columns:
            continue
        commanded = df_phantom[dof.phantom_col].values[mask]
        measured = np.interp(t_eval, t_etd, dof.etd_values(df_etd))
        diffs[dof.name] = measured - commanded

    return t_eval, diffs


# A run whose commanded origin does not coincide with the ETD's reference capture shows up as a
# near-constant offset between the two traces. Above this magnitude (mm / deg) the residual is
# dominated by that offset and the RMSE is not an accuracy figure.
REFERENCE_OFFSET_WARN = 1.0


def compute_dof_metrics(df_phantom, df_etd, window, dofs=None):
    """MAE / RMSE / max absolute error per DoF, inside the measurement window only.

    Also reports Reference_Offset, the median signed residual. The ETD reports displacement
    relative to its OWN reference surface capture, while the phantom trace is an absolute
    kinematic pose. For every run that starts at the kinematic home those two origins coincide and
    the median residual is ~0. For a run that does not - the vertical-slide series starts with the
    phantom already displaced, so its commanded longitudinal sits near -66 mm while the ETD reports
    around 0 - the residual is dominated by that static offset and the RMSE is NOT a tracking
    accuracy. Reference_Offset_Exceeded marks those rows instead of silently subtracting the
    offset, which would be the baseline zeroing this pipeline deliberately dropped.

    Returns a DataFrame with one row per DoF.
    """
    dofs = dofs or DOF_SPEC
    t_eval, diffs = _eval_grid(df_phantom, df_etd, window, dofs)
    t_start, t_end = window

    rows = []
    for dof in dofs:
        if dof.name not in diffs:
            continue
        diff = diffs[dof.name]
        offset = float(np.median(diff))
        rows.append({
            'DoF': dof.name,
            'Unit': dof.unit,
            'Mean_Absolute_Error': float(np.mean(np.abs(diff))),
            'RMSE': float(np.sqrt(np.mean(diff ** 2))),
            'Max_Absolute_Error': float(np.max(np.abs(diff))),
            'Reference_Offset': offset,
            'Reference_Offset_Exceeded': bool(abs(offset) > REFERENCE_OFFSET_WARN),
            'N_Samples': int(len(diff)),
            'Window_Start_Sec': t_start,
            'Window_End_Sec': t_end,
        })

    return pd.DataFrame(rows)


def compute_residual_vectors(df_phantom, df_etd, window, dofs=None):
    """Signed per-DoF residuals plus the 3D residual magnitudes, sample by sample.

    The 3D residual is the Euclidean norm of the deviation vector between commanded and measured
    pose, taken once over the three translational DoF and once over the three rotational DoF:

        d_trans(t) = || (dx, dy, dz) ||          [mm]
        d_rot(t)   = || (dpitch, dyaw, droll) || [deg]

    This is the quantity behind the optional third panel of the three-panel figure. It is a
    per-run quantity: averaging the SIGNED residuals of several runs first and taking the norm
    afterwards would let opposite-sign errors cancel and understate the deviation, so callers
    should compute this per run and average the magnitudes.

    Returns a DataFrame: Time_Sec, d_<dof> (signed) for each DoF, plus Resid3D_Trans / Resid3D_Rot.
    """
    dofs = dofs or DOF_SPEC
    t_eval, diffs = _eval_grid(df_phantom, df_etd, window, dofs)

    out = {'Time_Sec': t_eval}
    for name, diff in diffs.items():
        out[f'd_{name}'] = diff

    for label, triple in (('Resid3D_Trans', TRANSLATION_DOFS), ('Resid3D_Rot', ROTATION_DOFS)):
        present = [n for n in triple if n in diffs]
        if len(present) == len(triple):
            out[label] = np.sqrt(np.sum([diffs[n] ** 2 for n in triple], axis=0))
        else:
            missing = set(triple) - set(present)
            raise ValueError(f"Cannot form {label}: missing DoF {sorted(missing)}.")

    return pd.DataFrame(out)


def aggregate_runs(per_run_metrics, dofs=None):
    """Per-DoF RMSE mean and SD across the repeated runs of one group.

    per_run_metrics: list of DataFrames as returned by compute_dof_metrics, one per run.

    SD is the sample standard deviation (ddof=1) over the per-run RMSE values, i.e. the run-to-run
    reproducibility - this is what the error bars of the bar charts show. With the usual n = 3 runs
    it is a coarse estimate; N_Runs is reported alongside so that is visible.
    """
    dofs = dofs or DOF_SPEC
    frames = [m for m in per_run_metrics if m is not None and not m.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    rows = []
    for dof in dofs:
        sub = combined[combined['DoF'] == dof.name]
        if sub.empty:
            continue
        rmse = sub['RMSE'].values
        rows.append({
            'DoF': dof.name,
            'Unit': dof.unit,
            'RMSE_mean': float(np.mean(rmse)),
            # n = 1 has no run-to-run spread to estimate. NaN, not 0.0 - a zero would be plotted
            # as a vanishing error bar and read as perfect reproducibility.
            'RMSE_std': float(np.std(rmse, ddof=1)) if len(rmse) > 1 else float('nan'),
            'RMSE_min': float(np.min(rmse)),
            'RMSE_max': float(np.max(rmse)),
            'MAE_mean': float(np.mean(sub['Mean_Absolute_Error'].values)),
            'MaxAE_max': float(np.max(sub['Max_Absolute_Error'].values)),
            'N_Runs': int(len(rmse)),
            'N_Samples_total': int(sub['N_Samples'].sum()),
            # True if ANY run of this group has a commanded/ETD origin mismatch - see
            # compute_dof_metrics. Such a group's RMSE is a static offset, not an accuracy.
            'Reference_Offset_Exceeded': bool(sub['Reference_Offset_Exceeded'].any())
            if 'Reference_Offset_Exceeded' in sub.columns else False,
            'Reference_Offset_max': float(sub['Reference_Offset'].abs().max())
            if 'Reference_Offset' in sub.columns else float('nan'),
        })

    return pd.DataFrame(rows)


def combine_groups(per_run_metrics_by_group, dofs=None, weighting='pooled'):
    """One combined RMSE per DoF across several axis campaigns (e.g. Horizontal / Vertical /
    Rotation), so the talk needs one bar chart instead of three.

    per_run_metrics_by_group: {group_name: [per-run DataFrame, ...]}

    weighting:
      'pooled' (default) - sample-size weighted:

            RMSE_pooled = sqrt( sum_i N_i * RMSE_i^2 / sum_i N_i )

          taken over ALL individual runs i of all groups. This is exactly the RMSE one would
          obtain by concatenating every residual sample of every run, so it is the true combined
          RMSE rather than an average of averages.

    Two different spreads are reported, because they answer different questions and can differ by
    more than a factor of ten:

      SD_between_runs  sample-size weighted SD of all individual run RMSE values about the
                       combined value. Dominated by the genuine axis-to-axis dependence of the
                       error - i.e. "how much does the residual depend on which axis is moving".
                       This is the wide band.
      SD_within_axis   mean of the per-group run-to-run SDs. Pure measurement reproducibility of
                       a repeated run, with the axis dependence removed. This is the narrow cap.

      'inverse_variance' - each GROUP contributes with weight 1/SD_group^2 (SD over its own runs),
          combined as in a fixed-effect meta-analysis.

          CAVEAT, and the reason this is not the default: with n = 3 runs per group the SD itself
          carries roughly 50 % relative uncertainty, so the weights are extremely noisy and
          systematically favour whichever group happened to get a small SD. Worse, the three axis
          campaigns are not three noisy measurements of one common value - longitudinal error
          during a horizontal run and during a rotation run are genuinely different conditions,
          so the between-group spread is real signal, not sampling noise. The fixed-effect
          combined SD, 1/sqrt(sum w_i), then collapses far below any input SD and understates the
          uncertainty. Provided for comparison; reported in the intermediate CSV.

    Returns a DataFrame with one row per DoF.
    """
    dofs = dofs or DOF_SPEC
    rows = []

    for dof in dofs:
        runs = []  # (rmse, n_samples, group)
        for group, frames in per_run_metrics_by_group.items():
            for m in frames:
                if m is None or m.empty:
                    continue
                sub = m[m['DoF'] == dof.name]
                if sub.empty:
                    continue
                runs.append((float(sub['RMSE'].iloc[0]), int(sub['N_Samples'].iloc[0]), group))
        if not runs:
            continue

        rmse = np.array([r[0] for r in runs], dtype=float)
        n = np.array([r[1] for r in runs], dtype=float)
        groups = sorted({r[2] for r in runs})

        # Mean within-axis run-to-run SD (the narrow error bar), independent of the weighting.
        within = []
        for group in groups:
            gr = np.array([r[0] for r in runs if r[2] == group], dtype=float)
            if len(gr) > 1:
                within.append(float(np.std(gr, ddof=1)))
        # No group had repeated runs, so within-axis reproducibility is not measurable.
        # NaN rather than 0.0, which would assert perfect reproducibility.
        sd_within = float(np.mean(within)) if within else float('nan')

        if weighting == 'pooled':
            mean = float(np.sqrt(np.sum(n * rmse ** 2) / np.sum(n)))
            # Weighted sample SD of the per-run RMSE about the pooled value (reliability weights).
            var = np.sum(n * (rmse - mean) ** 2) / (np.sum(n) - np.sum(n ** 2) / np.sum(n)) \
                if len(rmse) > 1 else 0.0
            sd = float(np.sqrt(max(var, 0.0)))
            note = f"sample-weighted over {len(rmse)} runs / {len(groups)} axes, N = {int(n.sum())}"

        elif weighting == 'inverse_variance':
            g_mean, g_var, used_groups = [], [], []
            for group in groups:
                gr = np.array([r[0] for r in runs if r[2] == group], dtype=float)
                if len(gr) < 2:
                    continue
                v = float(np.var(gr, ddof=1))
                if v <= 0:
                    continue
                g_mean.append(float(np.mean(gr)))
                # Meta-analysis weights are 1/SE^2, not 1/SD^2: the quantity being combined is each
                # group's MEAN RMSE, whose standard error is SD/sqrt(n).
                g_var.append(v / len(gr))
                used_groups.append(group)
            if not g_var:
                continue
            w = 1.0 / np.array(g_var)
            mean = float(np.sum(w * np.array(g_mean)) / np.sum(w))
            sd = float(1.0 / np.sqrt(np.sum(w)))
            # Provenance must describe the groups that actually entered this estimate, not the
            # full input - groups with <2 runs or zero variance were skipped above.
            contributing = [r for r in runs if r[2] in used_groups]
            rmse = np.array([r[0] for r in contributing], dtype=float)
            n = np.array([r[1] for r in contributing], dtype=float)
            groups = used_groups
            # ...and so must the within-axis spread: a group dropped from the weights above must
            # not still pull the reported reproducibility around.
            within_used = [float(np.std([r[0] for r in runs if r[2] == g], ddof=1))
                           for g in used_groups
                           if len([r for r in runs if r[2] == g]) > 1]
            sd_within = float(np.mean(within_used)) if within_used else float('nan')
            note = f"inverse-variance over {len(g_var)} axes (see combine_groups docstring)"

        else:
            raise ValueError(f"Unknown weighting '{weighting}'.")

        rows.append({
            'DoF': dof.name,
            'Unit': dof.unit,
            'RMSE_combined': mean,
            'SD_between_runs': sd,
            'SD_within_axis': sd_within,
            'Weighting': weighting,
            'N_Runs': int(len(rmse)),
            'N_Samples_total': int(n.sum()),
            'Groups': ', '.join(groups),
            'Note': note,
        })

    return pd.DataFrame(rows)
