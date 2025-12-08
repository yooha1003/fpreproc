# scripts/run_sub007_preproc.py

## Overview
- Runs SingleSubjectPipeline for `sub-007` but skips connectivity and visualization steps, stopping after preprocessing.

## Usage
- `python scripts/run_sub007_preproc.py` (paths are fixed to `/data/data2/dataset/proc` and `/data/data2/dataset/fpreproc/results`).

## Inputs
- Raw data for `sub-007` under `/data/data2/dataset/proc/sub-007`.

## Outputs
- Preprocessing results for `sub-007` in `/data/data2/dataset/fpreproc/results/preprocessing/sub-007`.

## Notes
- Skip list excludes connectivity, ICA, EC, and visualization steps by default.
- Adjust the script or underlying pipeline call if different paths are required.
