# scripts/preprocessing/02_slice_timing.py

## Overview
- Applies slice timing correction using AFNI 3dTshift with a nilearn interpolation fallback.

## Usage
- `python scripts/preprocessing/02_slice_timing.py INPUT_MOCO OUTPUT_DIR [--subject ID] [--tr SEC] [--config FILE]`

## Inputs
- Motion-corrected fMRI volume, TR from config or flag, and slice order config under `preprocessing.slice_timing`.

## Outputs
- Slice timing corrected NIfTI plus metadata JSON noting TR, slice order, and AFNI tpattern; may be skipped if disabled in the config.

## Notes
- Uses AFNI when available; otherwise performs per-slice interpolation in Python.
- Returns the input unmodified when `slice_timing.enable` is false.
