# scripts/connectivity/functional_connectivity.py

## Overview
- Extracts ROI time series using a selected atlas and computes connectivity matrices plus graph metrics.

## Usage
- `python scripts/connectivity/functional_connectivity.py INPUT_FUNC OUTPUT_DIR [--subject ID] [--atlas NAME] [--config FILE]`

## Inputs
- Preprocessed functional image; config section `connectivity.functional` (methods, threshold, group_mode) and atlas settings.

## Outputs
- ROI time series `.npy`, connectivity matrices per method, graph metrics JSON, and metadata JSON in OUTPUT_DIR.

## Notes
- Supports correlation, partial correlation, tangent (group-mode only), covariance, and precision.
- Graph metrics use NetworkX and BCT; tangent is skipped for single-subject runs when group mode is off.
