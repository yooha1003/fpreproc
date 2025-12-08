# scripts/utils/quality_control.py

## Overview
- QC helpers for motion metrics, plots, temporal SNR computation, DVARS-based outlier detection, carpet plots, and HTML report generation.

## Usage
- Instantiate with an output directory (`qc = QualityControl('path')`) then call `compute_motion_metrics`, `plot_motion_parameters`, `compute_tsnr`, `detect_outliers`, `plot_carpet_plot`, or `create_qc_report`.

## Inputs
- Motion parameter arrays or nibabel images depending on the method; brain mask required for DVARS/outlier detection and carpet plots.

## Outputs
- Numeric metrics (dicts or arrays) and saved figures/HTML reports in the configured output directory.

## Notes
- Rotational parameters are converted to millimeters using a 50 mm radius for FD computations.
