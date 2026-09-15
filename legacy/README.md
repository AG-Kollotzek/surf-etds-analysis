# Legacy scripts

The evaluation chain that preceded the September 2026 re-evaluation. Kept for provenance and for
reproducing older results — **not** on the active path. Nothing in the current pipeline imports
anything from this folder.

The active chain lives in the repository root: `paper_pipeline.py`, `paper_plots.py`,
`qa_metrics.py`, `DataConverter.py`, `kinematics_werror_v2.py`, `xray_verification_v2.py`,
`export_bundle.py`.

## Running these

File contents are unchanged, so their imports still assume everything sits in one directory. Run
them from the **repository root** with both directories on the path:

```bash
PYTHONPATH=.:legacy python legacy/batch_evaluation.py
```

They also resolve `paper_data/` relative to the working directory, so the repository root is the
only place they work from.

## What is here, and why it was superseded

| File | Superseded by | Note |
|---|---|---|
| `kinematics.py` | `kinematics_werror_v2.py` | **Stale and wrong.** Its `h*cos(pitch)` term has the opposite sign to the whole `kinematics_werror*` line. On a pure horizontal sweep its correlation with the ETD trace is −0.9996 (RMSE 12.0 mm) against +0.9996 (0.18 mm) for v2. Do not use it as a reference curve. |
| `kinematics_werror.py` | `kinematics_werror_v2.py` | Correct signs, but uncalibrated — no fitted lever-arm offset and a small-angle rotation approximation. |
| `batch_evaluation.py` | `paper_pipeline.py` + `paper_plots.py` | The interactive three-panel plot editor. Still works; its zoom windows live in the root `zoom_box_config.json`, which is deliberately left in the root so this tool keeps finding it. The new pipeline uses `zoom_box_config_v2.json` instead — the two are not interchangeable, because the time origin moved from the crop start to the first sync pulse. |
| `batch_evaluation_old.py` | — | Earlier revision of the above. |
| `raw_data_export.py` | `paper_pipeline.py` | Wrote the old `paper_data/process_export/` tree. |
| `calculate_rmse.py` | `qa_metrics.py` | Plain per-DoF RMSE. Missing the ETD vertical sign correction and the evaluation window. |
| `export_metrics_csv.py` | `qa_metrics.aggregate_runs` | Produced `Paper_RMSD_Evaluation.csv` / `Paper_MaxDev_Evaluation.csv` using a different estimator (metric on the run-averaged curve). Cannot run as-is: it holds an absolute out-of-repo data path and loops over a group name that no longer exists. |
| `run_evaluation.py` | `paper_pipeline.py` | Broken — raises `NameError` on an undefined path. |
| `xray_verification_csv.py` | `xray_verification_v2.py` | **Actively broken now.** It hard-codes `Position = -10`, a leftover compensation for an earlier state of the model. Against the current model that pairs a sign-compensated input with an uncompensated model and yields a ~20 mm discrepancy. |
| `quick_plotter.py`, `evaluate_yaw_split.py`, `calculate_clin_phantompos.py`, `show_workspace3d.py` | — | One-off inspection tools. |

## `superseded_output/`

Results produced by the scripts above, kept only for comparison:

- `Paper_RMSD_Evaluation.csv`, `Paper_MaxDev_Evaluation.csv` — retired; use the per-run
  mean ± SD tables instead. Those are generated, not stored: run `python paper_pipeline.py` from
  the repository root and look under `paper_data/process_v2/`.
- `xray_verification_*.csv` — four generations of the X-ray table, all built with the
  uncalibrated kinematics. Superseded by `xray_verification_v2.csv` in the root.
- `paper_results.zip` — archive of the earlier figure set (the eight `first_try` PDFs).
- `paper_results_legacy.zip` — the 20 dated figure PDFs that used to sit in `paper_results/`.
  They are archived rather than kept loose because they cannot be regenerated: their producer,
  `batch_evaluation.py`, only writes a PDF when an operator types `save` into its interactive
  loop, and it stamps the current date into the output path.

The previous evaluation tree `paper_data/process_export/` is kept, tracked and unchanged, so the
old and new numbers can be compared directly.

**It cannot be regenerated.** It was written when `DataConverter` still imported the uncalibrated
`kinematics_werror`; the root `DataConverter` now imports `kinematics_werror_v2`, and
`raw_data_export.py` has no switch to select the old model. Re-running the legacy chain today
writes *calibrated* numbers into a tree that is supposed to document the pre-calibration state —
group RMSE differs by up to a factor of about six (RT/Vertical vertical: 0.077 mm committed vs
0.450 mm regenerated). To actually reproduce it you would have to pin `legacy/kinematics_werror.py`
back into `DataConverter`.
