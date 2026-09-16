# Changelog

## 1.0.0 — prepared for publication (September 2026)

- The raw data moved to the [`surf-etds-data`](https://github.com/AG-Kollotzek/surf-etds-data) repository,
  included as a submodule at the commit released as `v1.0-paper-2026`. `paper_pipeline.py` and `xray_verification_v2.py`
  read campaign `2026-03-10_L4` from there; the numbers do not change.
- The measurement protocol is published as a transcription in the data repository; the PDF is no longer here.
- Licence files, citation metadata, contributors and continuous integration added.
- Local absolute paths in `legacy/` replaced by `path/to/...` placeholders.
- `legacy/superseded_output/paper_results.zip` removed: it held the eight `first_try` figures together with
  macOS metadata files.

## Re-evaluation — September 2026

The evaluation was re-based on the calibrated kinematic model and the corrected numerical methodology of
`surf-etds-qa`. In order of how much each change moves the numbers:

1. **ETD vertical sign** handled in one place (`qa_metrics.DOF_SPEC`, `etd_sign = -1`).
2. **Calibrated kinematics** — lever-arm offset `SD_CALIB = 4.156 ± 0.550 mm` from the X-ray sphere
   detection of four linacs, and an exact rotation composition instead of a small-angle approximation.
3. **Alignment** — 50 % entry edges plus a least-squares time fit instead of a two-point anchor.
4. **No baseline zeroing** — the RMSE is the full residual.
5. **Evaluation window** — metrics restricted to the interval between the sync pulses.

Defects corrected while porting: the inverted `h·cos(pitch)` term in `legacy/kinematics.py`, and wrong pairing
of sync pulses when a terminal log holds more than two. Details in the [README](README.md#methodology-history)
and [`legacy/README.md`](legacy/README.md).

## Before September 2026

The earlier evaluation chain, its results and the reasons it was replaced are kept in [`legacy/`](legacy/README.md).
