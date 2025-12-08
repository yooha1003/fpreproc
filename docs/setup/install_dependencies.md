# setup/install_dependencies.py

## Overview
- Automates dependency checks and installs for the fMRI pipeline: neuroimaging software presence, pip packages, atlas downloads, and directory scaffolding.

## Usage
- `python setup/install_dependencies.py`

## Actions
- Verifies CLI availability of FSL, ANTs, AFNI, FreeSurfer, and optionally docker/fmriprep.
- Installs Python requirements from `requirements.txt`, upgrades pip, and confirms imports.
- Downloads common atlases via nilearn and builds the standard results/data/log directory layout.

## Outputs
- Console report of missing tools, package status, and atlas download attempts.
- Creates directory skeleton under the repo base; atlas files land in the nilearn cache/config paths.

## Notes
- No positional inputs are required; returns non-zero when installation or verification fails.
