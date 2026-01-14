# scripts/visualization/effective_connectivity_glass_brain_3d.py

## Overview
- BrainNet-like 3D directed effective connectivity visualization (HTML).
- Plots ROI nodes in MNI space and directed edges with arrowheads; optionally adds a translucent MNI brain surface.

## Usage
- `python scripts/visualization/effective_connectivity_glass_brain_3d.py MATRIX_NPY OUTPUT_DIR [--subject ID] [--method NAME] [--atlas NAME] [--top-k K] [--min-weight W] [--show-labels] [--roi-labels FILE] [--use-atlas-labels] [--node-size N] [--arrow-size MM] [--brain-opacity A] [--brain-step-size S] [--config FILE]`

## Inputs
- Directed EC matrix `.npy` using the repo convention: `matrix[target, source] = source → target`.
- Atlas selection for ROI coordinates (AAL, Schaefer, HarvardOxford).

## Outputs
- Interactive HTML saved in OUTPUT_DIR (default: `*_connectome_3d_directed.html`) plus a JSON metadata file.

## Notes
- Uses Plotly; output is self-contained HTML (no external CDN required).
- Edge selection is controlled via `--top-k` and `--min-weight` (by `|weight|`) to avoid clutter.
- Brain surface is generated from the MNI152 brain mask via marching cubes; increase `--brain-step-size` to speed up with lower detail.

