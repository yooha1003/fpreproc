# pipelines/single_subject_only.py

## Overview
- Wrapper around SingleSubjectPipeline that automatically removes group-only connectivity methods (e.g., tangent) while keeping the rest of the steps intact.

## Usage
- `python pipelines/single_subject_only.py SUBJECT_ID DATA_DIR OUTPUT_DIR [--config FILE] [--skip STEP ...] [--start-volume N]`

## Inputs
- Same subject folder layout and config options as `pipelines/single_subject.py`.

## Outputs
- Same per-subject folders and metadata as the standard pipeline, but without group-only functional connectivity methods.

## Notes
- Prints which connectivity methods were removed before running.
- Skip flags and start-volume overrides are forwarded unchanged.
