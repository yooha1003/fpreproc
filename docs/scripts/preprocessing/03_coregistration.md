# scripts/preprocessing/03_coregistration.py

## Overview
- Coregisters the functional mean image to anatomical space using FSL FLIRT, with ANTs or nilearn resampling as fallbacks.

## Usage
- `python scripts/preprocessing/03_coregistration.py FUNC_IMAGE ANAT_IMAGE OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- Preprocessed functional image, anatomical T1, and config under `registration.func_to_anat` (dof, cost function, preferred method).

## Outputs
- Mean functional, registered mean, transform matrix, QC overlay PNG, and metadata JSON in OUTPUT_DIR.

## Notes
- Creates a mean functional volume before registration.
- Tries the preferred method (`flirt` or `ants`), then falls back to alternatives and nilearn if tools are missing.
