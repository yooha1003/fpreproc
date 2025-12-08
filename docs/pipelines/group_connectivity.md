# pipelines/group_connectivity.py

## Overview
- Computes group-level connectivity matrices (e.g., tangent) from saved ROI time series produced by single-subject runs.

## Usage
- `python pipelines/group_connectivity.py [--subjects SUB ... | --subjects-file FILE] [--conn-dir DIR] [--output-dir DIR] [--config FILE] [--methods METHOD ...]`

## Inputs
- ROI time series files `<conn-dir>/<sub>/sub-xxx_roi_timeseries.npy` for chosen subjects.
- Optional list of methods (correlation, partial correlation, tangent, covariance, precision).

## Outputs
- Group mean/connectivity matrices per method under `output-dir/<method>/`, plus per-subject matrices and optional reference covariance files.
- Summary JSON `group_connectivity_summary.json` and logs.

## Notes
- Requires at least two subjects for tangent space embedding.
- Skips subjects missing ROI time series but records them in the summary.
