# scripts/run_connectivity_except_sub007.py

## Overview
- Runs connectivity, ICA, effective connectivity, and visualization for all preprocessed subjects except `sub-007`.

## Usage
- `python scripts/run_connectivity_except_sub007.py [--preproc-root DIR] [--output-root DIR] [--config FILE] [--subjects SUB ...] [--resume-missing-only]`

## Inputs
- Preprocessed outputs under `preproc-root` (expects `<sub>/<sub>_smoothed.nii.gz` with fallbacks to MNI or raw functional).
- Optional explicit subject list; otherwise discovers `sub-*` folders.

## Outputs
- Connectivity artifacts under `output-root/connectivity/<sub>` and visualizations under `output-root/visualization/<sub>`.
- Logs status per subject; returns non-zero if any fail.

## Notes
- Chooses the best available functional volume in priority order smoothed -> MNI -> raw.
- Skips existing outputs when `--resume-missing-only` is set.
