"""Paper / talk figures for the re-evaluated 2026-03-10 campaign.

Two families of figure:

  * RMSE bar charts   - per-DoF RMSE, mean +/- SD across the repeated runs of a group
                        (phantom validation: the all-axes sequence; scanner characterisation:
                        the isolated-axis campaigns), plus the single collapsed chart that
                        combines Horizontal / Vertical / Rotation into one weighted value.
  * Three-panel plots - see batch_evaluation.py; this module only supplies the shared style.

All labels are English. The palette and rcParams are the ones already used by
batch_evaluation.plot_evaluation_results_interactive, so every figure of the set matches.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qa_metrics as qm

# --- Okabe-Ito palette, identical to batch_evaluation.py -------------------------------------
C_X_PITCH = '#D55E00'   # vermillion  - lateral / pitch
C_Y_YAW = '#56B4E9'     # sky blue    - longitudinal / yaw
C_Z_ROLL = '#009E73'    # bluish green- vertical / roll
C_RESID_TRANS = '#CC79A7'
C_RESID_ROT = '#E69F00'

C_TRANSLATION = '#1B7C7C'   # teal, matching the bar colour of the current slides
C_ROTATION = '#7FB2B2'

RC_PARAMS = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 12, 'axes.edgecolor': 'black', 'axes.linewidth': 1.2,
    'legend.frameon': True, 'legend.edgecolor': 'black',
}


def apply_style():
    plt.rcParams.update(RC_PARAMS)


def _titles(ax, title, subtitle=None):
    """Bold title with an optional smaller grey second line, without the two colliding."""
    if subtitle:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=24)
        ax.text(0.5, 1.018, subtitle, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9.5, color='#444444')
    else:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)


def _dof_label(name, unit):
    label = qm.DOF_BY_NAME[name].label if name in qm.DOF_BY_NAME else name
    # "Longitudinal (Y)" -> "Longitudinal"; the axis letter is noise on a bar chart
    label = label.split(' (')[0]
    return f"{label} ({unit})"


def _order_rows(df):
    """Rows in DOF_SPEC order (translations first, then rotations)."""
    order = {d.name: i for i, d in enumerate(qm.DOF_SPEC)}
    out = df.copy()
    out['_o'] = out['DoF'].map(order)
    return out.sort_values('_o').drop(columns='_o').reset_index(drop=True)


def plot_rmse_bars(summary, title, out_path, subtitle=None, value_fmt='{:.2f}',
                   figsize=(8, 4.5), dpi=300):
    """Horizontal bar chart of per-DoF RMSE with the run-to-run SD as error bar.

    summary: DataFrame from qa_metrics.aggregate_runs
             (columns DoF, Unit, RMSE_mean, RMSE_std, N_Runs).
    """
    apply_style()
    df = _order_rows(summary)
    if df.empty:
        raise ValueError("empty summary - nothing to plot")

    labels = [_dof_label(r['DoF'], r['Unit']) for _, r in df.iterrows()]
    values = df['RMSE_mean'].values
    # NaN = not measurable (single run). np.nan_to_num would draw it as a zero-length bar, i.e.
    # claim perfect reproducibility; matplotlib skips NaN, which is the honest rendering.
    errors = df['RMSE_std'].values
    colors = [C_TRANSLATION if u == 'mm' else C_ROTATION for u in df['Unit']]

    y = np.arange(len(df))[::-1]  # first DoF at the top
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.barh(y, values, xerr=errors, color=colors, height=0.62,
            error_kw=dict(ecolor='black', capsize=3, lw=1.0))

    for yi, v, e in zip(y, values, errors):
        off = 0.0 if not np.isfinite(e) else e
        ax.text(v + off + 0.012 * max(values), yi, value_fmt.format(v),
                va='center', ha='left', fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('RMSE (mm / deg)')
    ax.set_xlim(0, float(np.nanmax(values + np.nan_to_num(errors))) * 1.25)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_runs = int(df['N_Runs'].iloc[0]) if 'N_Runs' in df.columns else None
    if n_runs is not None:
        run_note = (f"mean $\\pm$ SD over n = {n_runs} runs" if n_runs > 1
                    else "n = 1 run - no SD measurable")
        subtitle = f"{run_note} | {subtitle}" if subtitle else run_note
    _titles(ax, title, subtitle)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_combined_rmse_bars(combined, title, out_path, subtitle=None, value_fmt='{:.2f}',
                            figsize=(8.5, 4.8), dpi=300):
    """The single collapsed chart across the isolated-axis campaigns.

    Two spreads are drawn, because they answer different questions:
      * a light band  = SD_between_runs, the spread over all contributing runs. Wide, because the
                        residual genuinely depends on which axis is moving.
      * a capped bar  = SD_within_axis, the mean run-to-run reproducibility with the axis
                        dependence removed.

    combined: DataFrame from qa_metrics.combine_groups (weighting='pooled').
    """
    apply_style()
    df = _order_rows(combined)
    if df.empty:
        raise ValueError("empty combined table - nothing to plot")

    labels = [_dof_label(r['DoF'], r['Unit']) for _, r in df.iterrows()]
    values = df['RMSE_combined'].values
    sd_between = df['SD_between_runs'].values
    sd_within = df['SD_within_axis'].values
    colors = [C_TRANSLATION if u == 'mm' else C_ROTATION for u in df['Unit']]

    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.barh(y, values, height=0.58, color=colors, zorder=2)
    # Nested error bars: the wide grey one is the axis dependence, the thin black one the
    # run-to-run reproducibility. Drawn concentrically so the ratio between them is readable.
    ax.errorbar(values, y, xerr=sd_between, fmt='none', ecolor='#9A9A9A',
                elinewidth=6, capsize=0, alpha=0.85, zorder=3)
    ax.errorbar(values, y, xerr=sd_within, fmt='none', ecolor='black',
                elinewidth=1.2, capsize=4, zorder=4)

    right = float((values + sd_between).max())
    for yi, v, sb in zip(y, values, sd_between):
        ax.text(v + sb + 0.02 * right, yi, value_fmt.format(v),
                va='center', ha='left', fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('RMSE (mm / deg)')
    ax.set_xlim(0, right * 1.30)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _titles(ax, title, subtitle)

    n_runs = int(df['N_Runs'].max()) if 'N_Runs' in df.columns else None
    band = matplotlib.lines.Line2D(
        [0], [0], color='#9A9A9A', lw=6, alpha=0.85,
        label=(f'SD over all {n_runs} runs (axis dependence)' if n_runs
               else 'SD over all runs (axis dependence)'))
    cap = matplotlib.lines.Line2D([0], [0], color='black', lw=1.2, marker='|', markersize=8,
                                  label='mean within-axis SD (reproducibility)')
    ax.legend(handles=[band, cap], loc='lower right', fontsize=9, framealpha=0.95)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_temperature_comparison(temps, title, out_path, figsize=(9, 5), dpi=300):
    """Grouped bars RT vs 32 C per DoF, one panel per axis campaign."""
    apply_style()
    if temps.empty:
        raise ValueError("empty temperature table - nothing to plot")

    groups = list(dict.fromkeys(temps['Group']))
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, dpi=dpi, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, group in zip(axes, groups):
        sub = _order_rows(temps[temps['Group'] == group])
        x = np.arange(len(sub))
        ax.bar(x - 0.19, sub['RMSE_RT'], width=0.38, yerr=sub['SD_RT'],
               color=C_TRANSLATION, label='RT', error_kw=dict(ecolor='black', capsize=2, lw=0.9))
        ax.bar(x + 0.19, sub['RMSE_32C'], width=0.38, yerr=sub['SD_32C'],
               color=C_RESID_ROT, label='32 $\\degree$C', error_kw=dict(ecolor='black', capsize=2, lw=0.9))
        ax.set_xticks(x)
        ax.set_xticklabels([_dof_label(d, u) for d, u in zip(sub['DoF'], sub['Unit'])],
                           rotation=45, ha='right', fontsize=9)
        ax.set_title(group, fontsize=11)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('RMSE (mm / deg)')
    axes[0].legend(fontsize=9, loc='upper left')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


# =============================================================================================
#  Three-panel time-series figure
# =============================================================================================
#
#  Panel 1  translations  - commanded (dashed) vs ETD (solid), +/- SD band across runs
#  Panel 2  rotations     - same
#  Panel 3  selectable:
#             'etd_rms'    the tracker's OWN registration residuals as reported in the ETD log
#                          (fields rmse3D and rmseThermal). A scanner self-diagnostic, NOT an
#                          ETD-vs-phantom quantity and NOT a temperature.
#                          UNITS: deliberately not asserted on the axis. The two fields do not
#                          share a scale - measured over the campaign, rmse3D runs 0.78-1.93
#                          while rmseThermal runs 0.05-0.18 - and the ETD log documents neither.
#                          rmse3D is plausibly the surface-match residual in mm; rmseThermal is a
#                          separate thermal-camera quantity of unknown unit. Labelling the shared
#                          axis 'mm' would assert something the data contradicts.
#             'residual3d' the actual 3D residual between commanded and measured pose:
#                          |(dx,dy,dz)| in mm and |(dpitch,dyaw,droll)| in deg, computed per run
#                          and then averaged, with a +/- SD band.
#
#  Layout, palette and rcParams reproduce batch_evaluation.plot_evaluation_results_interactive.

PANEL3_MODES = ('etd_rms', 'residual3d')



def _draw_series(ax, t, mean, std, keys, colors, labels, line_w=1.2, band=True):
    """Dashed commanded + solid measured for one triple of DoF, with an optional SD band."""
    for key, color, label in zip(keys, colors, labels):
        ax.plot(t, mean['commanded'][key], color=color, linestyle='--', linewidth=line_w,
                label=f'Commanded {label}')
    for key, color, label in zip(keys, colors, labels):
        ax.plot(t, mean['etd'][key], color=color, linestyle='-', linewidth=line_w,
                label=f'ETD {label}')
        if band and std is not None:
            ax.fill_between(t, mean['etd'][key] - std['etd'][key], mean['etd'][key] + std['etd'][key],
                            color=color, alpha=0.2, linewidth=0)


def _add_zoom_inset(ax, spec, redraw, t):
    """One inset axis plus the marker rectangle and two crossing-free leader lines.

    redraw(axins) replots the panel content into the inset. Mirrors the inset behaviour of
    batch_evaluation.plot_evaluation_results_interactive.
    """
    pos = [float(v) for v in spec['pos']]
    xlims = tuple(float(v) for v in spec['xlim'])

    axins = ax.inset_axes(pos)
    # Opaque, and above the parent's artists: at 60 % alpha the evaluation-window shading
    # and the parent gridlines bled through and were easy to mistake for inset content.
    axins.set_facecolor('white')
    axins.set_zorder(5)
    axins.patch.set_alpha(1.0)
    redraw(axins)
    axins.set_xlim(xlims)

    ylims = spec.get('ylim')
    if ylims and len(ylims) == 2 and ylims[0] is not None and ylims[1] is not None:
        axins.set_ylim(float(ylims[0]), float(ylims[1]))
    else:
        mask = (t >= xlims[0]) & (t <= xlims[1])
        ys = [ln.get_ydata()[mask] for ln in axins.get_lines() if len(ln.get_ydata()) == len(t)]
        if ys:
            flat = np.concatenate(ys)
            flat = flat[np.isfinite(flat)]
            if flat.size:
                lo, hi = float(flat.min()), float(flat.max())
                margin = max(0.1, (hi - lo) * 0.15)
                axins.set_ylim(lo - margin, hi + margin)
    axins.set_xticklabels([])
    axins.tick_params(labelsize=8)

    y0, y1 = axins.get_ylim()
    rect = matplotlib.patches.Rectangle((xlims[0], y0), xlims[1] - xlims[0], y1 - y0,
                                        facecolor='none', edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    # Leader lines from the marked rectangle to the inset. Both the rectangle edge and the inset
    # corners are taken on the side that FACES the other, so the lines approach from outside and
    # never cross the inset's own plotting area. The inset normally sits above the data, so the
    # lines attach to its BOTTOM corners; if it is below, they attach to the top ones.
    to_axes = ax.transData + ax.transAxes.inverted()
    rect_centre_frac = to_axes.transform(((xlims[0] + xlims[1]) / 2.0, (y0 + y1) / 2.0))
    box_centre_x = pos[0] + pos[2] / 2.0
    box_centre_y = pos[1] + pos[3] / 2.0

    inset_is_above = box_centre_y > rect_centre_frac[1]
    corner_y = 0.0 if inset_is_above else 1.0          # inset edge facing the rectangle
    rect_y = y1 if inset_is_above else y0              # rectangle edge facing the inset

    if box_centre_x > rect_centre_frac[0]:
        pairs = ((xlims[0], 0.0), (xlims[1], 1.0))     # inset to the right: left->left, right->right
    else:
        pairs = ((xlims[1], 1.0), (xlims[0], 0.0))     # inset to the left

    for rect_x, corner_x in pairs:
        ax.add_artist(matplotlib.patches.ConnectionPatch(
            xyA=(rect_x, rect_y), xyB=(corner_x, corner_y),
            coordsA='data', coordsB='axes fraction', axesA=ax, axesB=axins,
            color='black', linewidth=0.8, alpha=0.6, zorder=4))
    return axins


def plot_three_panel(bundle, title, out_path, panel3='etd_rms', zoom_boxes=None,
                     figsize=(10, 12), dpi=300):
    """Non-interactive export of the three-panel figure.

    bundle: dict from paper_pipeline.combine_runs_for_plot with keys
            't', 'mean', 'std', 'etd_rms', 'residual', 'n_runs', 'window', 'lost_times'.
    panel3: 'etd_rms' or 'residual3d'.
    zoom_boxes: dict of zoom specs as stored in zoom_box_config.json, or None for no insets.
    """
    if panel3 not in PANEL3_MODES:
        raise ValueError(f"panel3 must be one of {PANEL3_MODES}, got {panel3!r}")

    apply_style()
    t = bundle['t']
    mean, std = bundle['mean'], bundle['std']
    line_w = 1.2

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, dpi=dpi)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])

    trans_keys = ('lateral', 'longitudinal', 'vertical')
    trans_colors = (C_X_PITCH, C_Y_YAW, C_Z_ROLL)
    trans_labels = ('X', 'Y', 'Z')
    rot_keys = ('pitch', 'yaw', 'roll')
    rot_colors = (C_X_PITCH, C_Y_YAW, C_Z_ROLL)
    rot_labels = ('pitch', 'yaw', 'roll')

    _draw_series(axes[0], t, mean, std, trans_keys, trans_colors, trans_labels, line_w)
    axes[0].set_ylabel('Translation (mm)')
    axes[0].legend(loc='upper right', ncol=2, fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    _draw_series(axes[1], t, mean, std, rot_keys, rot_colors, rot_labels, line_w)
    axes[1].set_ylabel('Rotation (deg)')
    axes[1].legend(loc='upper right', ncol=2, fontsize=9)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(min(lo, -0.5), max(hi, 0.5))

    # Reserve headroom at the top of panels 1 and 2 so neither the legend nor a zoom inset ever
    # sits on a trace. Insets occupy the upper ~35 % of the axes, the legend alone needs less.
    inset_panels = set()
    if zoom_boxes:
        for spec in zoom_boxes.values():
            if isinstance(spec, dict) and spec.get('active', True):
                try:
                    inset_panels.add(int(spec['ax_idx']))
                except (KeyError, TypeError, ValueError):
                    continue
    for idx, ax in ((0, axes[0]), (1, axes[1])):
        lo, hi = ax.get_ylim()
        keep = 0.55 if idx in inset_panels else 0.78
        ax.set_ylim(lo, lo + (hi - lo) / keep)

    if panel3 == 'etd_rms':
        axes[2].plot(t, bundle['etd_rms']['rmse3d'], color=C_RESID_TRANS, linewidth=line_w,
                     label='ETD surface-match RMS (3D)')
        axes[2].plot(t, bundle['etd_rms']['rmse_temp'], color=C_RESID_ROT, linewidth=line_w,
                     label='ETD thermal-camera RMS')
        # No unit on the axis - see the note at the top of this section.
        axes[2].set_ylabel('ETD-reported registration RMS')
    else:
        res = bundle['residual']
        tr = res['t']
        axes[2].plot(tr, res['trans_mean'], color=C_RESID_TRANS, linewidth=line_w,
                     label='3D residual, translation (mm)')
        axes[2].fill_between(tr, res['trans_mean'] - res['trans_std'],
                             res['trans_mean'] + res['trans_std'],
                             color=C_RESID_TRANS, alpha=0.2, linewidth=0)
        axes[2].plot(tr, res['rot_mean'], color=C_RESID_ROT, linewidth=line_w,
                     label='3D residual, rotation (deg)')
        axes[2].fill_between(tr, res['rot_mean'] - res['rot_std'],
                             res['rot_mean'] + res['rot_std'],
                             color=C_RESID_ROT, alpha=0.2, linewidth=0)
        axes[2].set_ylabel('3D position error (mm / deg)')
        axes[2].set_ylim(bottom=0)

    axes[2].set_xlabel('Time relative to first sync pulse (s)')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # Evaluation window, so it is visible which part of the record feeds the metrics.
    if bundle.get('window'):
        w0, w1 = bundle['window']
        for ax in axes:
            ax.axvspan(w0, w1, color='#000000', alpha=0.04, linewidth=0, zorder=0)

    # Tracking-lost intervals. The ETD logs nothing while tracking is lost, so every trace is
    # interpolated straight across the gap - the curve there is drawn, not measured. Marking them
    # keeps that visible (batch_evaluation.create_plot did the same; omitting it here would have
    # been a silent regression). Frames closer than 0.5 s are merged into one block, and a single
    # dropped frame is widened to 0.1 s so it stays visible.
    lost = sorted(set(bundle.get('lost_times') or []))
    if lost:
        blocks, start, prev = [], lost[0], lost[0]
        for t_lost in lost[1:]:
            if t_lost - prev > 0.5:
                blocks.append((start, prev))
                start = t_lost
            prev = t_lost
        blocks.append((start, prev))
        for i, (t0, t1) in enumerate(blocks):
            t1 = max(t1, t0 + 0.1)
            for ax in axes:
                ax.axvspan(t0, t1, color='#D55E00', alpha=0.18, linewidth=0, zorder=1,
                           label='ETD tracking lost' if i == 0 and ax is axes[0] else None)
        # Fold the marker into the panel-1 legend rather than adding a fourth legend box.
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc='upper right', ncol=2, fontsize=9)

    if zoom_boxes:
        redrawers = {
            0: lambda a: _draw_series(a, t, mean, std, trans_keys, trans_colors, trans_labels, line_w),
            1: lambda a: _draw_series(a, t, mean, std, rot_keys, rot_colors, rot_labels, line_w),
        }
        for name, spec in zoom_boxes.items():
            if not isinstance(spec, dict) or not spec.get('active', True):
                continue
            try:
                ax_idx = int(spec['ax_idx'])
            except (KeyError, TypeError, ValueError):
                continue
            if ax_idx not in redrawers:
                # Panel 3 holds a different quantity in each mode; an inset there would need its
                # own redraw branch. Skipped rather than drawn empty (the old code drew it empty).
                print(f"   [i] zoom box '{name}' targets panel {ax_idx + 1} - skipped "
                      f"(insets are supported on panels 1 and 2).")
                continue
            try:
                _add_zoom_inset(axes[ax_idx], spec, redrawers[ax_idx], t)
            except Exception as exc:
                print(f"   [!] zoom box '{name}' skipped: {exc}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path
