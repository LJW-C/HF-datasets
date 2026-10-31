# Revised Scientific Pipeline and Compatibility Notes

The root README is the primary guide for this GitHub checkout. Dataset version:
`2.0.0-rc1`; reserved identifier `10.5281/zenodo.23795076`. Publication state
was not independently verified at preparation. No data ZIP is rebuilt by this
repository update.

## Sources and Exactness

Core files were copied from the frozen `code_snapshot_2.0.0-rc1.zip`, not
reconstructed from the manuscript. Original archive paths and hashes appear in
`../data/frozen_source_mapping.json`. Current hashes and adaptations appear in
`../data/repository_source_mapping.json`.

- `channel.py`, `build_verified_dataset.py`, `source_checks.py`,
  `train_classical_baseline.py`, `test_pipeline.py`, `rfml_benchmark/data.py` and
  `rfml_benchmark/models.py` remain byte-identical to the frozen snapshot.
- `record_verified_sources.py` resolves root-level helper imports, adds
  `--list-devices`, and requires an explicit machine-specific `--device-id`.
  Audio packages are imported after parsing so `--help` works without hardware
  dependencies; actual capture still requires the declared audio packages.
  Mode selection, saved payload/control records, capture and QC algorithms are
  unchanged. Installing this code does not rerecord any sources.
- `train_suite.py` resolves root-level training sources and accepts
  `--device cuda|cpu|auto`. Original reference commands used CUDA. It records
  the requested device and preserves seed/model/quota/training defaults.
  Failed subprocesses surface their console-log tail for diagnosis.
- `plot_compact_examples.py` creates its output folder when missing. The
  signal representations, selected examples and PSD zooms are unchanged.
- The root general-purpose visualization script uses the noninteractive Agg
  backend for image export on headless hosts; it is not a new signal transform.
- `../train_rf_classifier.py` loads its locally created checkpoint safely on
  current PyTorch and claims a new output folder rather than overwriting runs.
  Epochs must be positive. Optimizer, model, scheduler, data order and metrics
  are unchanged; no published score was rerun for these compatibility edits.

The frozen snapshot remains authoritative for exact archived file bytes. A
portable GitHub commit is not retroactively inserted into its already frozen
manifest, and no edited file is described as byte-identical in the current
source mapping. Cross-platform bitwise training identity is not guaranteed.

## Entry Points

| Script | Purpose |
| --- | --- |
| `record_verified_sources.py` | Fldigi control readback, payload persistence and source screening |
| `build_verified_dataset.py` | Split-aware analytic conversion, corrected fading and provenance |
| `verify_dataset.py` | Fixed-index, raw-partition, guard, profile and finite-vector checks |
| `test_pipeline.py` | Synthetic pipeline and Doppler PSD regression tests |
| `validate_channel.py` | Long-process Gaussian PSD diagnostics; not ITU certification |
| `train_suite.py` | Six neural baselines with verified predefined splits |
| `train_classical_baseline.py` | 78-feature Random Forest baseline |
| `make_revision_results.py` | Export results/plots from explicitly selected run folders |
| `build_paired_window_study.py` | Separate paired four-configuration 2048/8192-point arrays |
| `check_paired_windows.py` | Verify matching controls and shared underlying noise |
| `run_duration_experiments.py` | Original fixed-layout repetition/duration runner |
| `summarize_supplementary_experiments.py` | Recompute repeat/duration metrics against predictions |
| `verify_reserved_upload.py` | Audit this exact DOI-reservation-time package, not live DOI state |

## Separate Duration Diagnostic

Extract the original benchmark and WAV ZIPs as documented at the root. Build
in a **new** output folder:

```powershell
python revision_v2/build_paired_window_study.py --out-root revision_v2/paired_windows_new --vectors-per-mode-snr 400 --seed 20261030
```

Train each arm through `train_suite.py` with `--models iq_cnn`, and through
`train_classical_baseline.py`, using seeds 20261029, 20261030, 20261031 and new
run folders. The historical convenience runner and summarizer expect the
original `paired_windows_20261030/`, `duration_runs_20261030/`,
`runs/seed_20261029/`, and `repeat_runs_20261030/` layouts. Do not point them
at frozen archived outputs and expect an overwrite or automatic resume.

This control contains only four configurations, 12,800 vectors per arm,
8960/1920/1920 split, common active starts and 8192-point guard. It does not
modify the 17-class benchmark. Its three-seed results cover IQ-CNN and Random
Forest, not Conformer. The main 17-class repeated runs cover Conformer and
Random Forest only. CSVs committed here are frozen measured results, not a
promise that another device or fresh source capture reproduces every score.
