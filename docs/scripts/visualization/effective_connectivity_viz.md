# scripts/visualization/effective_connectivity_viz.py

## Overview
- Visualizes directed effective connectivity matrices as circular arrow plots highlighting influence strength.

## Usage
- `python scripts/visualization/effective_connectivity_viz.py MATRIX_NPY OUTPUT_DIR [--subject ID] [--method NAME] [--top-k N] [--min-weight W] [--config FILE]`

## Inputs
- Directed effective connectivity matrix (`n_rois x n_rois`) with row=target and column=source; optional thresholds to limit drawn edges.

## Outputs
- Figure file (default PNG) stored in OUTPUT_DIR plus a small metadata dictionary from the `run` method.

## Notes
- Uses matplotlib; color scale maps to absolute weight magnitude.
- Restricts edges with `top_k` and `min_weight` to declutter dense matrices.
