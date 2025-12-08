# scripts/preprocessing/01_motion_correction.py

## Overview
- Performs motion correction using FSL MCFLIRT, falling back to AFNI 3dvolreg or a simplified nilearn realign routine.

## Usage
- `python scripts/preprocessing/01_motion_correction.py INPUT_FMRI OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- 4D fMRI image; pipeline config with `preprocessing.motion_correction` settings (cost function, reference volume, etc.).

## Outputs
- Motion-corrected NIfTI, motion parameter text file, QC plots, and metadata JSON in OUTPUT_DIR.
- Framewise displacement metrics are logged.

## Notes
- Prefers MCFLIRT; if unavailable or failing, tries 3dvolreg then nilearn.
- Uses `QualityControl` utilities for FD calculation and plotting.
