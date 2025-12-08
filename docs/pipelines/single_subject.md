# pipelines/single_subject.py

## Overview
- End-to-end pipeline for one subject: load data, motion correction, slice timing, coregistration, normalization, smoothing, functional connectivity, ICA, effective connectivity, and visualizations.

## Usage
- `python pipelines/single_subject.py SUBJECT_ID DATA_DIR OUTPUT_DIR [--config FILE] [--skip STEP ...] [--start-volume N]`

## Inputs
- SUBJECT_ID matching a folder under DATA_DIR with `anat/` and `func/` images.
- Config YAML controlling preprocessing and analysis parameters.

## Outputs
- Results in `OUTPUT_DIR/preprocessing|connectivity|visualization/SUBJECT_ID`.
- Metadata JSON `SUBJECT_ID_pipeline_results.json` plus per-step metadata and QC plots.

## Notes
- Uses the config-defined start volume when not provided; `--skip` can omit steps such as `ica` or `network_viz`.
- Logging is written to `OUTPUT_DIR/logs`.
