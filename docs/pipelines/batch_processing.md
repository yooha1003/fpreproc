# pipelines/batch_processing.py

## Overview
- BatchProcessor wraps SingleSubjectPipeline to run many subjects sequentially or in parallel.

## Usage
- `python pipelines/batch_processing.py DATA_DIR OUTPUT_DIR [--subjects ...] [--sequential] [--n-jobs N] [--skip STEP ...] [--start-volume N] [--config FILE]`

## Inputs
- Subject folders (`sub-*`) under DATA_DIR containing `anat/` and `func/` NIfTI data.
- Optional YAML config overriding defaults in `config/pipeline_config.yaml`.

## Outputs
- Per-subject preprocessing/connectivity results under OUTPUT_DIR plus `batch_processing_summary.json` with success/failure counts.
- Log files written to `OUTPUT_DIR/logs`.

## Notes
- Parallel mode uses ProcessPoolExecutor; set `--sequential` to disable.
- `--skip` and `--start-volume` are forwarded to SingleSubjectPipeline.
