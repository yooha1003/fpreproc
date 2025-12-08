# scripts/preprocessing/normalization.py

## Overview
- Wrapper that exposes `SpatialNormalization` from `04_normalization.py`.
- Functional images are now normalized with a full transform chain (func→anat→MNI) when a coregistration matrix is available.

## Usage
- `from preprocessing.normalization import SpatialNormalization`

## Notes
- Use the numbered script for CLI execution and detailed behavior.
- If a FLIRT `*_func2anat.mat` exists in the preprocessing directory, it is converted to ITK (via `c3d_affine_tool`) and applied together with the ANTs anat→MNI warp during functional normalization. If the coregistration was done with ANTs, the ITK transform is used as-is.
- Metadata now records both the original func→anat matrix and the converted ITK path (when generated).
