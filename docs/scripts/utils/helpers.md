# scripts/utils/helpers.py

## Overview
- Shared utilities: load configs, set up logging, save/load metadata, locate templates, compute masks and time series, estimate framewise displacement, and create registration overlays.

## Usage
- Import helpers such as `load_config`, `setup_logging`, `save_metadata`, `get_standard_template`, `plot_registration_overlay`, `estimate_framewise_displacement`, and `create_confound_regressors` from this module.

## Inputs/Outputs
- Functions accept paths or numpy/nibabel objects and return loaded configs, saved JSON files, masks, FD arrays, and QC figures.

## Notes
- Defaults to `config/pipeline_config.yaml` when no config path is provided.
- Includes FSL detection (`get_fsldir`) and brain template resolution via nilearn when FSL templates are missing.
