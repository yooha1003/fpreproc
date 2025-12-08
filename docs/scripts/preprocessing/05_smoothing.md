# scripts/preprocessing/05_smoothing.py

## Overview
- Performs spatial smoothing on normalized functional data via FSL `fslmaths`, AFNI `3dmerge`, or nilearn.

## Usage
- `python scripts/preprocessing/05_smoothing.py INPUT_FUNC OUTPUT_DIR [--subject ID] [--fwhm MM] [--config FILE]`

## Inputs
- Functional image in template space; smoothing kernel from config (`preprocessing.smoothing.fwhm`) or `--fwhm`.

## Outputs
- Smoothed NIfTI and metadata JSON noting FWHM, saved under OUTPUT_DIR.

## Notes
- Converts FWHM to sigma for FSL; sequentially falls back to AFNI then nilearn if tools are missing.
