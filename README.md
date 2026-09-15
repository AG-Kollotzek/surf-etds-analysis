# surf-etds-analysis

Evaluation pipeline for the tracking accuracy of a surface-guided radiotherapy (SGRT) system.
A motorised phantom — the SURF test unit — is driven along known trajectories while an
ExacTrac Dynamic (ETD) surface scanner tracks it; this repository compares what the scanner
measured against what the phantom was commanded to do, over all six degrees of freedom.

It turns the ESTRO-ACROP Table 4 (D2) requirement *"tracking performance: 1 mm / 1°"* into an
auditable number. Everything here runs from the raw measurement logs, which are included in full.

> **Part of a two-repository methodology.** This repository covers the **analysis**. The phantom
> hardware, its firmware and the SURF logging terminal live in a separate repository — see
> [Related repositories](#related-repositories).

---

## Results at a glance

Campaign of 2026-03-10, room temperature, combined over the three isolated-axis series
(Horizontal, Vertical, Rotation), sample-weighted over 9 runs, N = 6641 samples:

| Degree of freedom | RMSE | Tolerance (ESTRO-ACROP D2) |
|---|---|---|
| Longitudinal | 0.149 mm | 1 mm |
| Lateral | 0.225 mm | 1 mm |
| Vertical | 0.088 mm | 1 mm |
| Pitch | 0.055° | 1° |
| Yaw | 0.094° | 1° |
| Roll | 0.070° | 1° |

Phantom validation, the all-axes sequence (±10 mm horizontal and vertical, ±5° rotation),
mean ± SD over 3 repeated runs: longitudinal 0.164 ± 0.028 mm, lateral 0.229 ± 0.010 mm,
vertical 0.100 ± 0.035 mm, pitch 0.046 ± 0.010°, yaw 0.090 ± 0.002°, roll 0.058 ± 0.009°.

Read [How to read the numbers](#how-to-read-the-numbers) before quoting any of these — several sit
close to the measurement's own detection limit, and the lateral value is not what it looks like.

---

## Reproduce it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

python paper_pipeline.py        # 32 runs -> tables + 50 figures in paper_data/process_v2/  (~1 min)
python xray_verification_v2.py  # independent radiographic check -> xray_verification_v2.csv
python export_bundle.py         # optional: one sorted ZIP for handover
```

Run from the repository root — every path is relative to it. The raw logs in `paper_data/full_raw/`
(36 MB) must be present, so clone the repository properly; a shallow clone or a "Download ZIP" of
the code alone will not work.

The table above is `paper_data/process_v2/05_weighted_rmse_RT.csv` after the first command, so you
can check your run reproduces it.

---

## What is generated, and why it is not stored here

**`paper_data/process_v2/` is not in the repository.** Every table and figure is a function of the
raw logs plus this code, and rebuilding the whole tree takes about a minute. Storing 13 MB of
derived PDFs would only add a second copy that can fall out of step with the code.

If a path mentioned below is missing after cloning, run `python paper_pipeline.py`.

Two generated things *are* kept on purpose: `xray_verification_v2.csv`, because it is the published
X-ray table; and the retired tables in `legacy/superseded_output/`, because the scripts that
produced them no longer reproduce their numbers.

---

## What is in this repository

| Path | |
|---|---|
| `paper_pipeline.py` | **Start here.** Processes every run, writes all tables, renders all figures. |
| `paper_plots.py` | The figures: RMSE bar charts and the three-panel time-series plots. |
| `qa_metrics.py` | Metric definitions — MAE / RMSE / MaxAE, 3D residuals, aggregation across runs and axes. |
| `DataConverter.py` | Loads both logs, finds the sync pulses, aligns the two clocks, applies the kinematics. Also holds `MEASUREMENT_10032026`, the map from raw filename to run, group and pad state. |
| `kinematics_werror_v2.py` | Calibrated forward kinematics of the phantom, with uncertainty propagation. |
| `xray_verification_v2.py` | Independent radiographic check of the kinematic model against X-ray sphere detection. |
| `export_bundle.py` | Packs tables, figures and the X-ray table into one sorted ZIP for handover. Run the pipeline first. |
| `paper_data/full_raw/` | **The measurement.** 38 ETD tracking JSON + 38 ETD preview PNG + 40 SURF terminal CSV. |
| `Messung_Protokoll_10_03_2026.pdf` | Measurement protocol. The only source for the hand-transcribed X-ray readouts in `xray_verification_v2.py`. |
| `zoom_box_config_v2.json` | Zoom-inset windows for the figures. Auto-detected when absent; edit by hand to override. |
| `zoom_box_config.json` | Belongs to `legacy/batch_evaluation.py`; kept in the root so that tool still finds it. |
| `paper_data/process_export/` | Derived tree of the **previous** evaluation, kept as the pre-calibration comparison baseline. Not regenerable — see `legacy/README.md`. |
| `legacy/` | The superseded evaluation chain, with its own README explaining what replaced what — and which three files are not merely old but wrong to reuse. |

---

## The dataset

One campaign, 2026-03-10, on an Elekta Versa HD. 32 evaluated runs in six groups — all axes,
horizontal, vertical, rotation, variable speed, vertical slide — each recorded at room temperature
and with the phantom's surface heated to 32 °C. `paper_data/full_raw/` additionally contains setup
and X-ray logs that are not part of the 32 runs.

Two systems log independently, with unsynchronised PC clocks: the SURF terminal writes the
commanded axis positions, the ETD writes its measured pose. A 5 mm sync pulse at the start and end
of every run is the only shared physical event, and it anchors the alignment.

Which raw file belongs to which run is recorded in `MEASUREMENT_10032026` in `DataConverter.py`.

---

## How the analysis works

1. **Load** both logs — ETD tracking JSON and SURF terminal CSV (`DataConverter`).
2. **Kinematics** — convert commanded axis positions into a clinical 6-DoF pose through the
   calibrated forward model (`kinematics_werror_v2`).
3. **Align** — locate the sync pulses by value stability, anchor on their 50 % entry edges, then
   fit time scale and offset by least squares against the whole longitudinal curve to remove the
   drift between the two clocks (`DataConverter`).
4. **Window** — restrict everything that follows to the interval between the sync pulses, with a
   3 s margin, so the sync movement itself never enters the statistics (`qa_metrics`).
5. **Compare** — residual = measured − commanded, per degree of freedom, plus the 3D residual
   magnitude over the translation and rotation triples (`qa_metrics`).
6. **Aggregate** — RMSE per run, then mean ± SD across the repeated runs of a group, then a
   sample-weighted combination across axis campaigns (`qa_metrics`), and render (`paper_plots`).

No baseline zeroing: the reported RMSE is the full residual, static offset included.

---

## How to read the numbers

1. **Detection limit.** The ETD logs shifts quantised to 0.1 mm / 0.1°, so the
   quantisation-limited RMSE floor is 0.1/√12 ≈ **0.029**. Values within a factor of two or three
   of that are at the detection limit, not resolved differences.
2. **Lateral RMSE is not a tracking error.** The kinematic model constrains lateral motion to
   exactly zero, yet X-ray sphere detection independently measures a −0.5/+0.6 mm lateral
   excursion at a vertical deflection of ±10 mm. The lateral figure is therefore dominated by a
   real, radiographically confirmed cross-coupling of the vertical axis — not by the scanner
   failing to track.
3. **"All axes" is sequential.** Horizontal, then vertical, then rotation, returning to zero
   between. It is not a combined multi-axis motion.
4. **Effective sample size.** The phantom logs at a steady 9.2–9.4 Hz; the ETD rate varies
   (7.5–14.5 Hz) and is *slower* than the phantom in 15 of the 32 runs. There the ETD is
   up-sampled onto the phantom grid, so successive residuals are correlated and the sample count
   overstates the number of independent measurements. This affects confidence intervals derived
   from N, not the RMSE itself.
5. **The thermal effect is not resolvable.** Heating the surface to 32 °C changes the per-DoF RMSE
   by at most 0.056 mm / 0.028°, the sign is not consistent across axes, and every difference is
   below roughly 2.4σ of the run-to-run scatter at n = 3.

---

## Validation

Guards that stop a wrong number passing silently. Each is exercised by the campaign:

- **Commanded/ETD origin mismatch.** The ETD zeroes its shifts on its own reference capture, so a
  run that does not start at the kinematic home carries a static offset rather than a tracking
  error. Flagged per DoF, warned about, and excluded from the combined figures — the vertical-slide
  series would otherwise report a 66 mm "RMSE".
- **Offset-invariant time refinement.** The alignment fits the shape, not the level, so a static
  bias cannot masquerade as clock drift.
- **Automatic sync-pair selection.** A terminal log can be longer than the ETD scan and hold more
  than two sync pulses. The pair is chosen by span match and rejected beyond 5 % deviation.
- **Undefined statistics stay undefined.** A single-run group reports SD as `NaN`, never `0.0`.
- **Tracking dropouts.** Counted, reported, and shaded on the figures, because the residual is
  interpolated across them rather than measured.

---

## Related repositories

| | |
|---|---|
| **Hardware, firmware, logging terminal** | Phantom mechanics and axes, motor control, the SURF terminal that writes the commanded-position logs, and the log-format specification. *Repository to be published — the link will appear here.* |
| **This repository** | Synchronisation, kinematic model, metric definitions, statistics, figures. |
| `surf-etds-qa` | The clinical, multi-linac QA application of the same phantom, with per-linac reports. Separate repository; its report machinery is deliberately **not** part of this one. |

The two repositories together form the methodology: the hardware repository defines how the
measurement is produced and what the log columns mean; this one defines how it is evaluated.

---

## Methodology history

The evaluation was re-based in September 2026 on the calibrated kinematic model and the corrected
numerical methodology from `surf-etds-qa`. In order of how much each change moves the numbers:

1. **ETD vertical sign** handled in one place (`qa_metrics.DOF_SPEC`, `etd_sign = -1`). The previous
   chain treated it inconsistently between its RMSE and its plotting paths.
2. **Calibrated kinematics** — fitted lever-arm offset `SD_CALIB = 4.156 ± 0.550 mm`, calibrated
   against the X-ray sphere detection of four linacs, and an exact rotation composition replacing a
   small-angle approximation.
3. **Alignment** — 50 % entry edges plus a least-squares time fit, replacing a two-point anchor.
4. **No baseline zeroing** — the RMSE is now the full residual.
5. **Evaluation window** — metrics restricted to the interval between the sync pulses.

Two defects were found and corrected while porting: `legacy/kinematics.py` carries an inverted
`h·cos(pitch)` term relative to the rest of the model line, and a terminal log holding more than two
sync pulses could be paired wrongly without any error. Both are described in `legacy/README.md`.

---

## Kurzfassung (DE)

Auswertungspipeline für die Tracking-Genauigkeit eines Oberflächenscanners (SGRT). Ein motorisiertes
Phantom fährt bekannte Trajektorien, der ExacTrac-Dynamic-Scanner verfolgt es; dieses Repository
vergleicht Soll und Ist über alle sechs Freiheitsgrade und macht aus der ESTRO-ACROP-Vorgabe
„1 mm / 1°" eine überprüfbare Zahl. Rohdaten sind vollständig enthalten, alle Tabellen und Plots
werden mit `python paper_pipeline.py` neu erzeugt.
