# scripts/visualization/glass_brain_network.py

## Overview
- Generates multiple visualizations (glass brain, 3D HTML, circular chord, heatmap) from a connectivity matrix.

## Usage
- `python scripts/visualization/glass_brain_network.py MATRIX_NPY OUTPUT_DIR [--subject ID] [--atlas NAME] [--config FILE]`

## Inputs
- Connectivity matrix `.npy`; atlas selection for ROI coordinates (AAL, Schaefer, HarvardOxford).

## Outputs
- PNGs and HTML saved in OUTPUT_DIR: glass brain connectome, interactive 3D connectome, circular plot, matrix heatmap, plus metadata JSON.

## Notes
- Uses nilearn plotting; trims or pads when coordinate count and matrix size differ.
- Plotly is optional; 3D output is skipped if the library is missing.
