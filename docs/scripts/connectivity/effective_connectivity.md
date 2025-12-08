# scripts/connectivity/effective_connectivity.py

## Overview
- Computes directed effective connectivity from ROI time series via Granger causality, transfer entropy, and optional spectral Granger analysis.

## Usage
- `python scripts/connectivity/effective_connectivity.py TIMESERIES_NPY OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- ROI time series `.npy` produced by the functional connectivity step; config `connectivity.effective` for methods and lag settings.

## Outputs
- Numpy matrices for selected methods (granger, transfer_entropy, spectral) and metadata JSON describing shapes and methods.

## Notes
- Relies on statsmodels for Granger tests and mne-connectivity for spectral GC when available; falls back silently if packages are missing.
