# scripts/visualization/effective_connectivity_source_sink_viz.py

## Overview
- Computes node-level EC source/sink metrics (in-strength, out-strength, net-flow) and generates a summary figure + CSV.

## Usage
- `python scripts/visualization/effective_connectivity_source_sink_viz.py MATRIX_NPY OUTPUT_DIR [--subject ID] [--method NAME] [--top-n N] [--top-k K] [--min-weight W] [--use-abs] [--roi-labels FILE] [--config FILE]`

## Inputs
- Directed EC matrix `.npy` using the repo convention: `matrix[target, source] = source → target`.
- Optional ROI label text file (one label per line).

## Outputs
- A multi-panel PNG (default) showing top sources/sinks, out-vs-in scatter, and a sorted EC heatmap.
- A CSV with per-ROI metrics (`*_node_metrics.csv`) and a JSON metadata file.

## Notes
- `--top-k`/`--min-weight` apply edge filtering before computing metrics (by `|weight|`).
- `--use-abs` computes metrics on absolute weights (useful if your EC method produces signed values).

