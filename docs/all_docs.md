# fpreproc Documentation (All-in-One)

This file concatenates every Markdown doc under `docs/` with their original paths.

---

## pipelines/batch_processing.md

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


---

## pipelines/group_connectivity.md

# pipelines/group_connectivity.py

## Overview
- Computes group-level connectivity matrices (e.g., tangent) from saved ROI time series produced by single-subject runs.

## Usage
- `python pipelines/group_connectivity.py [--subjects SUB ... | --subjects-file FILE] [--conn-dir DIR] [--output-dir DIR] [--config FILE] [--methods METHOD ...]`

## Inputs
- ROI time series files `<conn-dir>/<sub>/sub-xxx_roi_timeseries.npy` for chosen subjects.
- Optional list of methods (correlation, partial correlation, tangent, covariance, precision).

## Outputs
- Group mean/connectivity matrices per method under `output-dir/<method>/`, plus per-subject matrices and optional reference covariance files.
- Summary JSON `group_connectivity_summary.json` and logs.

## Notes
- Requires at least two subjects for tangent space embedding.
- Skips subjects missing ROI time series but records them in the summary.


---

## pipelines/single_subject.md

# pipelines/single_subject.py

## Overview
- End-to-end pipeline for one subject: load data, motion correction, slice timing, coregistration, normalization, smoothing, functional connectivity, ICA, effective connectivity, and visualizations.

## Usage
- `python pipelines/single_subject.py SUBJECT_ID DATA_DIR OUTPUT_DIR [--config FILE] [--skip STEP ...] [--start-volume N]`

## Inputs
- SUBJECT_ID matching a folder under DATA_DIR with `anat/` and `func/` images.
- Config YAML controlling preprocessing and analysis parameters.

## Outputs
- Results in `OUTPUT_DIR/preprocessing|connectivity|visualization/SUBJECT_ID`.
- Metadata JSON `SUBJECT_ID_pipeline_results.json` plus per-step metadata and QC plots.

## Notes
- Uses the config-defined start volume when not provided; `--skip` can omit steps such as `ica` or `network_viz`.
- Logging is written to `OUTPUT_DIR/logs`.


---

## pipelines/single_subject_only.md

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


---

## scripts/connectivity/effective_connectivity.md

# scripts/connectivity/effective_connectivity.py

## Overview
- Computes directed effective connectivity from ROI time series via Granger causality, transfer entropy, and optional spectral Granger analysis.

## Usage
- `python scripts/connectivity/effective_connectivity.py TIMESERIES_NPY OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- ROI time series `.npy` produced by the functional connectivity step; config `connectivity.effective` for methods and lag settings.

## Outputs
- Numpy matrices for selected methods (granger, transfer_entropy, spectral) and metadata JSON describing shapes and methods.

## Notes
- Relies on statsmodels for Granger tests and mne-connectivity for spectral GC when available; falls back silently if packages are missing.


---

## scripts/connectivity/functional_connectivity.md

# scripts/connectivity/functional_connectivity.py

## Overview
- Extracts ROI time series using a selected atlas and computes connectivity matrices plus graph metrics.

## Usage
- `python scripts/connectivity/functional_connectivity.py INPUT_FUNC OUTPUT_DIR [--subject ID] [--atlas NAME] [--config FILE]`

## Inputs
- Preprocessed functional image; config section `connectivity.functional` (methods, threshold, group_mode) and atlas settings.

## Outputs
- ROI time series `.npy`, connectivity matrices per method, graph metrics JSON, and metadata JSON in OUTPUT_DIR.

## Notes
- Supports correlation, partial correlation, tangent (group-mode only), covariance, and precision.
- Graph metrics use NetworkX and BCT; tangent is skipped for single-subject runs when group mode is off.


---

## scripts/connectivity/ica_analysis.md

# scripts/connectivity/ica_analysis.py

## Overview
- Runs ICA (CanICA) on functional data, saves spatial components and time courses, classifies components, and notes a DMN index when possible.

## Usage
- `python scripts/connectivity/ica_analysis.py INPUT_FUNC OUTPUT_DIR [--subject ID] [--n-components N] [--config FILE]`

## Inputs
- Preprocessed functional image; `connectivity.ica` config for component count and algorithm hints.

## Outputs
- ICA component image, time courses `.npy`, spatial weights, component classifications JSON, and metadata JSON.

## Notes
- Uses nilearn CanICA with EPI masking; simple heuristics label components as signal or noise.
- Includes placeholder DMN detection; dual regression helper is available for downstream analyses.


---

## scripts/preprocessing/01_motion_correction.md

# scripts/preprocessing/01_motion_correction.py

## Overview
- Performs motion correction using FSL MCFLIRT, falling back to AFNI 3dvolreg or a simplified nilearn realign routine.

## Usage
- `python scripts/preprocessing/01_motion_correction.py INPUT_FMRI OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- 4D fMRI image; pipeline config with `preprocessing.motion_correction` settings (cost function, reference volume, etc.).

## Outputs
- Motion-corrected NIfTI, motion parameter text file, QC plots, and metadata JSON in OUTPUT_DIR.
- Framewise displacement metrics are logged.

## Notes
- Prefers MCFLIRT; if unavailable or failing, tries 3dvolreg then nilearn.
- Uses `QualityControl` utilities for FD calculation and plotting.


---

## scripts/preprocessing/02_slice_timing.md

# scripts/preprocessing/02_slice_timing.py

## Overview
- Applies slice timing correction using AFNI 3dTshift with a nilearn interpolation fallback.

## Usage
- `python scripts/preprocessing/02_slice_timing.py INPUT_MOCO OUTPUT_DIR [--subject ID] [--tr SEC] [--config FILE]`

## Inputs
- Motion-corrected fMRI volume, TR from config or flag, and slice order config under `preprocessing.slice_timing`.

## Outputs
- Slice timing corrected NIfTI plus metadata JSON noting TR, slice order, and AFNI tpattern; may be skipped if disabled in the config.

## Notes
- Uses AFNI when available; otherwise performs per-slice interpolation in Python.
- Returns the input unmodified when `slice_timing.enable` is false.


---

## scripts/preprocessing/03_coregistration.md

# scripts/preprocessing/03_coregistration.py

## Overview
- Coregisters the functional mean image to anatomical space using FSL FLIRT, with ANTs or nilearn resampling as fallbacks.

## Usage
- `python scripts/preprocessing/03_coregistration.py FUNC_IMAGE ANAT_IMAGE OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- Preprocessed functional image, anatomical T1, and config under `registration.func_to_anat` (dof, cost function, preferred method).

## Outputs
- Mean functional, registered mean, transform matrix, QC overlay PNG, and metadata JSON in OUTPUT_DIR.

## Notes
- Creates a mean functional volume before registration.
- Tries the preferred method (`flirt` or `ants`), then falls back to alternatives and nilearn if tools are missing.


---

## scripts/preprocessing/04_normalization.md

# scripts/preprocessing/04_normalization.py

## Overview
- Normalizes anatomical and functional images to MNI space using FSL (FLIRT + FNIRT) or ANTs workflows, with a nilearn resampling fallback.

## Usage
- `python scripts/preprocessing/04_normalization.py ANAT_IMAGE FUNC_IMAGE OUTPUT_DIR [--subject ID] [--config FILE]`

## Inputs
- Subject anatomical and functional images; config `registration.anat_to_standard` controlling method (`fnirt` or `ants`), template, and ANTs parameters.

## Outputs
- Normalized anatomical and functional NIfTIs, warp or affine transforms, QC overlay, and metadata JSON in OUTPUT_DIR.

## Notes
- Fetches template via `utils.helpers.get_standard_template`.
- Applies the computed transform to functional data; tries multiple backends before falling back to resampling.


---

## scripts/preprocessing/05_smoothing.md

# scripts/preprocessing/05_smoothing.py

## Overview
- Performs spatial smoothing on normalized functional data via FSL `fslmaths`, AFNI `3dmerge`, or nilearn.

## Usage
- `python scripts/preprocessing/05_smoothing.py INPUT_FUNC OUTPUT_DIR [--subject ID] [--fwhm MM] [--config FILE]`

## Inputs
- Functional image in template space; smoothing kernel from config (`preprocessing.smoothing.fwhm`) or `--fwhm`.

## Outputs
- Smoothed NIfTI and metadata JSON noting FWHM, saved under OUTPUT_DIR.

## Notes
- Converts FWHM to sigma for FSL; sequentially falls back to AFNI then nilearn if tools are missing.


---

## scripts/preprocessing/__init__.md

# scripts/preprocessing/__init__.py

## Overview
- Marks the preprocessing directory as a package for pipeline imports.

## Notes
- Contains only a namespace docstring; processing code lives in the numbered modules.


---

## scripts/preprocessing/coregistration.md

# scripts/preprocessing/coregistration.py

## Overview
- Thin wrapper to expose `Coregistration` from `03_coregistration.py`.

## Usage
- `from preprocessing.coregistration import Coregistration`

## Notes
- For CLI execution and full options use `scripts/preprocessing/03_coregistration.py`.


---

## scripts/preprocessing/motion_correction.md

# scripts/preprocessing/motion_correction.py

## Overview
- Lightweight wrapper that exposes `MotionCorrection` from `01_motion_correction.py` for clean imports within the pipeline.

## Usage
- `from preprocessing.motion_correction import MotionCorrection`

## Notes
- The CLI and implementation live in `scripts/preprocessing/01_motion_correction.py`.


---

## scripts/preprocessing/normalization.md

# scripts/preprocessing/normalization.py

## Overview
- Wrapper that exposes `SpatialNormalization` from `04_normalization.py`.
- Functional images are now normalized with a full transform chain (func→anat→MNI) when a coregistration matrix is available.

## Usage
- `from preprocessing.normalization import SpatialNormalization`

## Notes
- Use the numbered script for CLI execution and detailed behavior.
- If a FLIRT `*_func2anat.mat` exists in the preprocessing directory, it is converted to ITK (via `c3d_affine_tool`) and applied together with the ANTs anat→MNI warp during functional normalization. If the coregistration was done with ANTs, the ITK transform is used as-is.
- Metadata now records both the original func→anat matrix and the converted ITK path (when generated).


---

## scripts/preprocessing/slice_timing.md

# scripts/preprocessing/slice_timing.py

## Overview
- Wrapper module that loads `SliceTimingCorrection` from `02_slice_timing.py`.

## Usage
- `from preprocessing.slice_timing import SliceTimingCorrection`

## Notes
- CLI execution and processing details reside in `scripts/preprocessing/02_slice_timing.py`.


---

## scripts/preprocessing/smoothing.md

# scripts/preprocessing/smoothing.py

## Overview
- Wrapper that re-exports `SpatialSmoothing` from `05_smoothing.py` for pipeline imports.

## Usage
- `from preprocessing.smoothing import SpatialSmoothing`

## Notes
- Invoke the numbered script for CLI usage and parameter overrides.


---

## scripts/run_all_except_sub007.md

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


---

## scripts/run_connectivity_except_sub007.md

# scripts/run_connectivity_except_sub007.py

## Overview
- Runs connectivity, ICA, effective connectivity, and visualization for all preprocessed subjects except `sub-007`.

## Usage
- `python scripts/run_connectivity_except_sub007.py [--preproc-root DIR] [--output-root DIR] [--config FILE] [--subjects SUB ...] [--resume-missing-only]`

## Inputs
- Preprocessed outputs under `preproc-root` (expects `<sub>/<sub>_smoothed.nii.gz` with fallbacks to MNI or raw functional).
- Optional explicit subject list; otherwise discovers `sub-*` folders.

## Outputs
- Connectivity artifacts under `output-root/connectivity/<sub>` and visualizations under `output-root/visualization/<sub>`.
- Logs status per subject; returns non-zero if any fail.

## Notes
- Chooses the best available functional volume in priority order smoothed -> MNI -> raw.
- Skips existing outputs when `--resume-missing-only` is set.


---

## scripts/run_sub007_preproc.md

# scripts/run_sub007_preproc.py

## Overview
- Runs SingleSubjectPipeline for `sub-007` but skips connectivity and visualization steps, stopping after preprocessing.

## Usage
- `python scripts/run_sub007_preproc.py` (paths are fixed to `/data/data2/dataset/proc` and `/data/data2/dataset/fpreproc/results`).

## Inputs
- Raw data for `sub-007` under `/data/data2/dataset/proc/sub-007`.

## Outputs
- Preprocessing results for `sub-007` in `/data/data2/dataset/fpreproc/results/preprocessing/sub-007`.

## Notes
- Skip list excludes connectivity, ICA, EC, and visualization steps by default.
- Adjust the script or underlying pipeline call if different paths are required.


---

## scripts/utils/check_ants_install.md

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


---

## scripts/utils/data_loader.md

# scripts/utils/data_loader.py

## Overview
- Loads subject anatomical and functional NIfTI data, trims initial volumes, concatenates 3D series to 4D, and validates folder structure.

## Usage
- `from utils.data_loader import NiftiDataLoader`; call `load_fmri_data`, `load_anatomical_data`, `get_subject_list`, or `validate_subject_data`. Running the module directly prints validation summaries.

## Inputs
- Base data directory containing `sub-*/anat` and `sub-*/func` files; start volume index (default 7) to drop initial volumes.

## Outputs
- nibabel images and metadata dicts describing shapes, voxel sizes, and volume counts; validation results with errors or warnings.

## Notes
- Handles both single 4D files and sorted 3D volume series; raises clear errors when expected files are missing.


---

## scripts/utils/helpers.md

# scripts/utils/helpers.py

## Overview
- Shared utilities: load configs, set up logging, save/load metadata, locate templates, compute masks and time series, estimate framewise displacement, and create registration overlays.

## Usage
- Import helpers such as `load_config`, `setup_logging`, `save_metadata`, `get_standard_template`, `plot_registration_overlay`, `estimate_framewise_displacement`, and `create_confound_regressors` from this module.

## Inputs/Outputs
- Functions accept paths or numpy/nibabel objects and return loaded configs, saved JSON files, masks, FD arrays, and QC figures.

## Notes
- Defaults to `config/pipeline_config.yaml` when no config path is provided.
- Includes FSL detection (`get_fsldir`) and brain template resolution via nilearn when FSL templates are missing.


---

## scripts/utils/quality_control.md

# scripts/utils/quality_control.py

## Overview
- QC helpers for motion metrics, plots, temporal SNR computation, DVARS-based outlier detection, carpet plots, and HTML report generation.

## Usage
- Instantiate with an output directory (`qc = QualityControl('path')`) then call `compute_motion_metrics`, `plot_motion_parameters`, `compute_tsnr`, `detect_outliers`, `plot_carpet_plot`, or `create_qc_report`.

## Inputs
- Motion parameter arrays or nibabel images depending on the method; brain mask required for DVARS/outlier detection and carpet plots.

## Outputs
- Numeric metrics (dicts or arrays) and saved figures/HTML reports in the configured output directory.

## Notes
- Rotational parameters are converted to millimeters using a 50 mm radius for FD computations.


---

## scripts/visualization/activation_patterns.md

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


---

## scripts/visualization/effective_connectivity_viz.md

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


---

## scripts/visualization/glass_brain_network.md

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


---

## setup/install_dependencies.md

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
