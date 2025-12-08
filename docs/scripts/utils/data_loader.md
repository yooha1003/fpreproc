# scripts/utils/data_loader.py

## Overview
- Loads subject anatomical and functional NIfTI data, trims initial volumes, concatenates 3D series to 4D, and validates folder structure.

## Usage
- `from utils.data_loader import NiftiDataLoader`; call `load_fmri_data`, `load_anatomical_data`, `get_subject_list`, or `validate_subject_data`. Running the module directly prints validation summaries.

## Inputs
- Base data directory containing `sub-*/anat` and `sub-*/func` files; start volume index (default 7) to drop initial volumes.

## Outputs
- nibabel images and metadata dicts describing shapes, voxel sizes, and volume counts; validation results with errors or warnings.

## Notes
- Handles both single 4D files and sorted 3D volume series; raises clear errors when expected files are missing.
