# scripts/visualization/activation_patterns.py

## Overview
- Creates activation and QC visualizations: mean functional image, tSNR map, optional ICA component maps, montage, and saves a tSNR volume.

## Usage
- `python scripts/visualization/activation_patterns.py INPUT_FUNC OUTPUT_DIR [--subject ID] [--ica-components FILE] [--config FILE]`

## Inputs
- Preprocessed functional image; optional ICA components image; visualization settings under `visualization.activation`.

## Outputs
- PNGs for mean functional, tSNR map, montage, and optional ICA components; tSNR NIfTI; metadata JSON with mean tSNR and file paths.

## Notes
- Uses QualityControl to compute tSNR; limits ICA plots to the first 10 components by default.
