"""Bundle the re-evaluated campaign into one sorted ZIP for handover.

Everything produced by paper_pipeline.py and xray_verification_v2.py is collected and sorted by
what it answers, not by where it happened to be written:

    01_RMSE/                per-DoF error statistics - the bar charts and every table behind them
    02_Residual_3D/         the 3D residual over time - figures and per-run time series
    03_Tracking_overview/   the same three-panel figures with the scanner's own registration RMS
    04_Xray_verification/   independent radiographic check of the kinematic model
    05_Aligned_source_data/ the aligned, kinematics-applied traces every number derives from

Per-run files are renamed to be self-describing (<temperature>_<group>_meas_NN_...), so a file
still says what it is once it has been dragged out of its folder.

Usage:
    python export_bundle.py                 # -> SURF_ETD_Evaluation_<date>.zip
    python export_bundle.py --out /tmp/x.zip
"""

import argparse
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path

SOURCE = Path('paper_data/process_v2')
XRAY_CSV = Path('xray_verification_v2.csv')

# Groups whose RMSE is a valid tracking accuracy. Others are exported but kept out of the
# headline folder - see the flags written by paper_pipeline.
ACCURACY_GROUPS = ('All_axes', 'Horizontal', 'Vertical', 'Rotation')


def _run_tag(meas_dir):
    """'RT_Horizontal_meas_04' from .../process_v2/RT/Horizontal/meas_04."""
    return f"{meas_dir.parent.parent.name}_{meas_dir.parent.name}_{meas_dir.name}"


def _group_tag(group_dir):
    """'RT_Horizontal' from .../process_v2/RT/Horizontal."""
    return f"{group_dir.parent.name}_{group_dir.name}"


def build_tree(root, source=SOURCE, verbose=True):
    """Lay the sorted tree out under `root`. Returns the number of files copied."""
    n = 0

    def copy(src, dst):
        nonlocal n
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1

    # --- 01 RMSE ------------------------------------------------------------------------------
    rmse = root / '01_RMSE'
    for fig in sorted((source / 'figures').glob('fig_*.pdf')):
        copy(fig, rmse / 'figures' / fig.name)
    for csv in sorted(source.glob('05_weighted_rmse_*.csv')):
        copy(csv, rmse / '01_combined_across_axes' / csv.name)
    for csv in sorted(source.glob('06_temperature_comparison.csv')):
        copy(csv, rmse / '01_combined_across_axes' / csv.name)
    for csv in sorted(source.glob('*/*/04_group_rmse_summary.csv')):
        copy(csv, rmse / '02_per_group' / f"{_group_tag(csv.parent)}_group_rmse_summary.csv")
    for csv in sorted(source.glob('*/*/meas_*/03_alldof_rmse.csv')):
        copy(csv, rmse / '03_per_run' / f"{_run_tag(csv.parent)}_alldof_rmse.csv")

    # --- 02 / 03 the three-panel figures, split by which quantity panel 3 carries --------------
    for sub, label in (('with_zoom', 'figures_with_zoom'), ('no_zoom', 'figures_no_zoom')):
        for fig in sorted((source / 'figures' / sub).glob('*.pdf')):
            target = '02_Residual_3D' if 'residual3d' in fig.name else '03_Tracking_overview'
            copy(fig, root / target / label / fig.name)

    for csv in sorted(source.glob('*/*/meas_*/03_residual3d.csv')):
        copy(csv, root / '02_Residual_3D' / 'per_run_timeseries' /
             f"{_run_tag(csv.parent)}_residual3d.csv")

    # --- 04 X-ray -----------------------------------------------------------------------------
    if XRAY_CSV.exists():
        copy(XRAY_CSV, root / '04_Xray_verification' / XRAY_CSV.name)

    # --- 05 aligned source traces -------------------------------------------------------------
    for meas in sorted(source.glob('*/*/meas_*')):
        tag = _run_tag(meas)
        for name in ('02_after_align_etd.csv', '02_after_align_phantom.csv', 'metadata.json'):
            src = meas / name
            if src.exists():
                copy(src, root / '05_Aligned_source_data' / tag / name)

    if verbose:
        print(f"[+] {n} files laid out under {root.name}/")
    return n


def collect_flags(source=SOURCE):
    """Runs the pipeline flagged, so the README can name them explicitly."""
    offset, lost = [], []
    for meta_path in sorted(source.glob('*/*/meas_*/metadata.json')):
        meta = json.loads(meta_path.read_text())
        tag = _run_tag(meta_path.parent)
        if meta.get('tracking_lost_in_window', 0):
            lost.append((tag, meta['tracking_lost_in_window']))
    for csv in sorted(source.glob('*/*/04_group_rmse_summary.csv')):
        import pandas as pd
        df = pd.read_csv(csv, sep=';')
        if 'Reference_Offset_Exceeded' in df.columns and df['Reference_Offset_Exceeded'].any():
            offset.append((_group_tag(csv.parent), float(df['Reference_Offset_max'].max())))
    return offset, lost


def write_readme(root, source=SOURCE):
    offset_flagged, lost_flagged = collect_flags(source)

    offset_txt = '\n'.join(
        f"  - {g}: static offset up to {v:.1f} mm/deg" for g, v in offset_flagged) or "  - none"
    lost_txt = '\n'.join(
        f"  - {t}: {n} frame(s)" for t, n in lost_flagged) or "  - none"

    text = f"""SURF / ExacTrac Dynamic - tracking evaluation
Measurement campaign 2026-03-10, re-evaluated {date.today().isoformat()}

===============================================================================
WHAT THIS IS
===============================================================================
An ExacTrac Dynamic (ETD) surface scanner tracks a motorised phantom whose
commanded pose is known from a calibrated forward-kinematic model of its three
axes. Every number here is a comparison of the two: measured (ETD) against
commanded (phantom), over six degrees of freedom.

  Translations  longitudinal, lateral, vertical   [mm]
  Rotations     pitch, yaw, roll                  [deg]

===============================================================================
FOLDER GUIDE
===============================================================================
01_RMSE/                    Per-DoF error statistics.
  figures/                  Bar charts. fig_allaxes_* is the phantom-validation
                            result; fig_<axis>_* the individual axis campaigns;
                            fig_combined_* collapses those three into one
                            sample-weighted value per DoF; fig_temperature_*
                            compares room temperature against 32 deg C.
  01_combined_across_axes/  The combined and temperature tables.
  02_per_group/             RMSE mean +/- SD over the repeated runs of a group.
  03_per_run/               MAE / RMSE / MaxAE for each individual run.

02_Residual_3D/             The 3D residual between commanded and measured pose:
                            |(dx,dy,dz)| in mm and |(dpitch,dyaw,droll)| in deg.
  figures_*/                Three-panel figures, panel 3 = 3D residual.
  per_run_timeseries/       The underlying per-sample residuals.

03_Tracking_overview/       The same three-panel figures, but panel 3 carries the
                            scanner's OWN registration residuals as logged by the
                            ETD (rmse3D, rmseThermal). That is a scanner
                            self-diagnostic, not an ETD-vs-phantom quantity.

04_Xray_verification/       Independent radiographic check. The phantom sphere was
                            imaged by stereoscopic kV at static deflections; this
                            compares those positions against the kinematic model,
                            without involving surface tracking at all.

05_Aligned_source_data/     The aligned, kinematics-applied traces every number
                            above derives from, plus per-run metadata (sync
                            timestamps, evaluation window, alignment parameters).

Every figure exists twice: with zoom insets and without (figures_with_zoom /
figures_no_zoom). Insets always span at least +/- 1 mm / 1 deg, the ESTRO-ACROP
Table 4 (D2) acceptance tolerance, so a deviation is read against the limit it
has to meet rather than magnified out of proportion. They are placed where the
measured and commanded traces actually separate - on the sub-millimetre
cross-coupling of the axes that were not driven, and on a commanded hold - not
on a large excursion, which would simply rescale to look like the main panel.

Shaded bands on the figures: light grey = the evaluation window that feeds the
metrics; orange = an interval in which the ETD reported tracking lost, where the
curve is interpolated across the gap rather than measured.

===============================================================================
METHOD
===============================================================================
Kinematics    Empirically calibrated lever arm (fitted against the X-ray sphere
              detection of four linacs) and an exact rotation composition.
Alignment     Sync pulses located by value stability and anchored on their 50 %
              entry edges, followed by a least-squares refinement of time scale
              and offset against the full longitudinal curve, which removes the
              drift between the two independent PC clocks. No baseline zeroing:
              the reported RMSE is the full residual, static offset included.
Window        Metrics are taken strictly between the sync pulses (3 s margin), so
              the sync movement and its settling do not enter the statistics.
Estimator     RMSE is computed per run; the mean +/- SD across the repeated runs
              of a group is what the error bars show. The combined chart pools by
              sample count, which is the RMSE of all residual samples taken
              together, and draws two spreads: the wide one is the genuine
              axis-to-axis dependence, the narrow one is run-to-run
              reproducibility.

===============================================================================
HOW TO READ THE NUMBERS - PLEASE NOTE
===============================================================================
1. DETECTION LIMIT. The ETD logs shifts quantised to 0.1 mm / 0.1 deg, giving a
   quantisation-limited RMSE floor of 0.1/sqrt(12) = 0.029. Values within a
   factor of two or three of that are at the detection limit, not resolved.

2. LATERAL IS NOT A TRACKING ERROR. The kinematic model constrains lateral
   motion to exactly zero, yet the X-ray sphere detection independently measures
   a -0.5 / +0.6 mm lateral excursion when the vertical axis is at +/-10 mm. The
   lateral RMSE is therefore dominated by a real, radiographically confirmed
   cross-coupling of the vertical axis that the model does not describe - not by
   a failure of the scanner to track.

3. THE ALL-AXES SEQUENCE IS SEQUENTIAL. Horizontal, then vertical, then
   rotation, returning to zero between. It is not a combined multi-axis motion.

4. EFFECTIVE SAMPLE SIZE. The phantom logs at a steady 9.2-9.4 Hz while the ETD
   rate varies between 7.5 and 14.5 Hz, and is slower than the phantom in 15 of
   the 32 runs. There the ETD is up-sampled onto the phantom grid, so successive
   residuals are correlated and the sample count overstates the number of
   independent measurements. This affects confidence intervals derived from N,
   not the RMSE itself.

5. THERMAL EFFECT IS NOT RESOLVABLE. Heating the phantom surface to 32 deg C
   changes the per-DoF RMSE by at most 0.056 mm / 0.028 deg, the sign is not
   consistent across axes, and every difference is below roughly 2.4 sigma of the
   run-to-run scatter at n = 3.

6. GROUPS WITH A COMMANDED/ETD ORIGIN MISMATCH. The ETD zeroes its shift values
   on its own reference surface capture. A run that does not begin at the
   kinematic home therefore carries a large static offset between the two traces,
   and its RMSE is that offset rather than a tracking error. Flagged, and
   excluded from every combined figure:
{offset_txt}

7. TRACKING DROPOUTS. Where the ETD reports tracking lost it logs no data, so the
   residual is interpolated across the gap rather than measured. Runs affected
   inside their evaluation window:
{lost_txt}

===============================================================================
REPRODUCING
===============================================================================
    python paper_pipeline.py        # all tables and figures
    python xray_verification_v2.py  # X-ray verification table
    python export_bundle.py         # this bundle

Repository: surf-etds-analysis. Methodology and the defects corrected during this
re-evaluation are documented in its README.
"""
    (root / '00_README.txt').write_text(text)
    return offset_flagged, lost_flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None, help="output .zip path")
    ap.add_argument('--source', default=str(SOURCE), help="pipeline output root")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"{source} not found - run paper_pipeline.py first.")

    stamp = date.today().isoformat()
    name = f"SURF_ETD_Evaluation_{stamp}"
    out_zip = Path(args.out) if args.out else Path(f"{name}.zip")

    staging = Path('.export_staging') / name
    if staging.parent.exists():
        shutil.rmtree(staging.parent)
    staging.mkdir(parents=True)

    build_tree(staging, source)
    write_readme(staging, source)

    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(staging.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(staging.parent))

    shutil.rmtree(staging.parent)
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[>] {out_zip}  ({size_mb:.1f} MB)")

    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
    print(f"    {len(names)} files")
    tops = sorted({n.split('/')[1] for n in names if n.count('/') >= 1 and '/' in n[len(name):].lstrip('/')})
    for t in sorted({n.split('/')[1] for n in names if len(n.split('/')) > 1}):
        cnt = sum(1 for n in names if n.split('/')[1] == t)
        print(f"      {t:26s} {cnt:4d} files")


if __name__ == '__main__':
    main()
