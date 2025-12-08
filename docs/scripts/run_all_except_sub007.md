# scripts/run_all_except_sub007.py

## Overview
- Convenience runner that batches every subject except `sub-007` through the pipeline using BatchProcessor.

## Usage
- `python scripts/run_all_except_sub007.py [--data-dir DIR] [--output-dir DIR] [--config FILE] [--sequential] [--n-jobs N] [--skip STEP ...] [--start-volume N]`

## Inputs
- Data directory with subject folders; optional config path.

## Outputs
- Preprocessing and analysis results in the given output directory; exit code 1 if any subject fails.

## Notes
- Discovers subjects via NiftiDataLoader and filters out `sub-007`.
- `--skip` passes through to the pipeline; `--sequential` disables multiprocessing.
