# scripts/utils/check_ants_install.py

## Overview
- Diagnostic tool to inspect ANTs installation, check binary locations and versions, and optionally run a test `antsRegistrationSyN.sh` call.

## Usage
- `python scripts/utils/check_ants_install.py [--json-report FILE] [--skip-versions] [--fixed IMG --moving IMG --output-prefix PREFIX] [--dimensionality N] [--transform TYPE] [--threads N] [--histogram-matching 0|1] [--keep-outputs]`

## Inputs
- Optional fixed and moving images to trigger a test registration; otherwise only environment checks are performed.

## Outputs
- Console report summarizing PATH/ANTSPATH, binary discovery, version strings, and optional test-run outputs; JSON report if requested.

## Notes
- Cleans up test outputs unless `--keep-outputs` is set.
- Returns structured dictionaries from helper functions for programmatic use.
