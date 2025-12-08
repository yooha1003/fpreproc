# scripts/connectivity/ica_analysis.py

## Overview
- Runs ICA (CanICA) on functional data, saves spatial components and time courses, classifies components, and notes a DMN index when possible.

## Usage
- `python scripts/connectivity/ica_analysis.py INPUT_FUNC OUTPUT_DIR [--subject ID] [--n-components N] [--config FILE]`

## Inputs
- Preprocessed functional image; `connectivity.ica` config for component count and algorithm hints.

## Outputs
- ICA component image, time courses `.npy`, spatial weights, component classifications JSON, and metadata JSON.

## Notes
- Uses nilearn CanICA with EPI masking; simple heuristics label components as signal or noise.
- Includes placeholder DMN detection; dual regression helper is available for downstream analyses.
