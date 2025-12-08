# scripts/preprocessing/04_normalization.py

## Overview
- Normalizes anatomical and functional images to MNI space using FSL (FLIRT + FNIRT) or ANTs workflows, with a nilearn resampling fallback.

## Usage
- `python scripts/preprocessing/04_normalization.py ANAT_IMAGE FUNC_IMAGE OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- Subject anatomical and functional images; config `registration.anat_to_standard` controlling method (`fnirt` or `ants`), template, and ANTs parameters.

## Outputs
- Normalized anatomical and functional NIfTIs, warp or affine transforms, QC overlay, and metadata JSON in OUTPUT_DIR.

## Notes
- Fetches template via `utils.helpers.get_standard_template`.
- Applies the computed transform to functional data; tries multiple backends before falling back to resampling.
