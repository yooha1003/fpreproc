# fMRI Preprocessing and Network Analysis Pipeline

A comprehensive, modular pipeline for preprocessing resting-state fMRI data and performing connectivity analysis, with a focus on epilepsy research.

## Features

### Preprocessing
- **Brain Extraction**: ANTs (soft/hard), deepbet, FSL BET, nilearn fallback
- **Motion Correction**: FSL MCFLIRT / AFNI 3dvolreg
- **Slice Timing Correction**: AFNI 3dTshift with explicit `-tpattern` support
- **Spatial Registration**: Functional to anatomical (FLIRT, ANTs SyN)
- **Normalization**: MNI standard space (FNIRT/ANTs SyN)
- **Spatial Smoothing**: Gaussian smoothing (FSL/AFNI)

### Connectivity Analysis
- **Functional Connectivity (FC)**
  - Pearson correlation
  - Partial correlation
  - Tangent space embedding (set `group_mode` for multi-subject runs)
  - Graph theory metrics (degree, clustering, modularity, etc.)

- **Independent Component Analysis (ICA)**
  - Automatic ICA decomposition
  - Component classification (signal/noise)
  - Default Mode Network (DMN) identification

- **Effective Connectivity (EC)**
  - Granger Causality
  - Transfer Entropy
  - Spectral Granger Causality

### Visualization
- **3D Glass Brain Networks**
  - Interactive connectome visualization
  - Circular connectome (chord diagram)
  - Connectivity matrix heatmaps

- **Activation Patterns**
  - tSNR maps
  - ICA component maps
  - Surface-based visualization

- **Directed Effective Connectivity**
  - Arrow-based plots for Granger / transfer entropy influence patterns

### Pipeline Execution
- **Single Subject**: Process one subject end-to-end
- **Batch Processing**: Parallel or sequential processing of multiple subjects
- **Modular Design**: Skip or customize any pipeline step
- **Python API**: Importable wrappers for each preprocessing stage (`scripts/preprocessing/*.py`)

## Directory Structure

```
fpreproc/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── config/
│   ├── pipeline_config.yaml          # Main configuration file
│   └── atlas/                        # Brain atlases
│
├── setup/
│   └── install_dependencies.py       # Dependency installer
│
├── scripts/
│   ├── preprocessing/                # Preprocessing modules
│   │   ├── 01_motion_correction.py
│   │   ├── 02_slice_timing.py
│   │   ├── 03_coregistration.py
│   │   ├── 04_normalization.py
│   │   └── 05_smoothing.py
│   │
│   ├── connectivity/                 # Connectivity analysis
│   │   ├── functional_connectivity.py
│   │   ├── ica_analysis.py
│   │   └── effective_connectivity.py
│   │
│   ├── visualization/                # Visualization tools
│   │   ├── glass_brain_network.py
│   │   ├── activation_patterns.py
│   │   ├── effective_connectivity_viz.py
│   │   ├── effective_connectivity_source_sink_viz.py
│   │   └── effective_connectivity_glass_brain_3d.py
│   │
│   ├── utils/                        # Utility functions
│   │   ├── data_loader.py
│   │   ├── helpers.py
│   │   └── quality_control.py
│   │
│   ├── run_all_except_sub007.py           # Batch helper excluding sub-007
│   ├── run_connectivity_except_sub007.py  # Connectivity/ICA/EC (except sub-007)
│   └── run_sub007_preproc.py              # Preprocess-only helper for sub-007
│
├── pipelines/
│   ├── single_subject.py             # Single subject pipeline
│   └── batch_processing.py           # Batch processing
│
├── data/                             # Your data goes here
│   └── raw/
│       └── sub-{ID}/
│           ├── anat/                 # T1 images (hdr/img)
│           └── func/                 # fMRI images (hdr/img)
│
└── results/                          # Output directory
    ├── preprocessing/
    ├── connectivity/
    ├── visualization/
    └── logs/
```

## Installation

### Prerequisites

The pipeline requires the following neuroimaging software (pre-installed on your system):
- **FSL** (FMRIB Software Library)
- **ANTs** (Advanced Normalization Tools)
- **AFNI** (Analysis of Functional NeuroImages)
- **FreeSurfer** (optional)
- **fMRIPrep** (optional)

### Python Environment Setup

1. **Clone or navigate to the repository**

2. **Install Python dependencies**

```bash
# Using pip
pip install -r requirements.txt

# OR using conda
conda env create -f environment.yml
conda activate fmri_pipeline
```

3. **Run dependency installer**

```bash
python setup/install_dependencies.py
```

This will:
- Check for neuroimaging software
- Install Python packages
- Download brain atlases
- Create directory structure
- Verify imports

## Data Organization

Place your data in the following structure:

### Option 1: Single 4D NIfTI File per Subject (Recommended)
```
data/raw/
├── sub-001/
│   ├── anat/
│   │   └── T1.nii.gz
│   └── func/
│       └── func.nii.gz
├── sub-002/
│   └── ...
└── sub-003/
    └── ...
```

### Option 2: 3D NIfTI Series (Multiple Files per Subject)
```
data/raw/
├── sub-001/
│   ├── anat/
│   │   └── T1.nii.gz
│   └── func/
│       ├── func_0007.nii.gz
│       ├── func_0008.nii.gz
│       └── ... (multiple 3D volumes)
├── sub-002/
│   └── ...
```

**Important Notes**:
- The pipeline automatically detects and handles both formats
- **NIfTI format** (`.nii.gz` or `.nii`) is required
- For **3D series**: Files are sorted by volume number (e.g., `_0007`, `_0008`) and concatenated into 4D
- The **first 6 volumes** are automatically removed (configurable: `fmri_start_volume: 7` in `config/pipeline_config.yaml`)
- Anatomical file can be named `T1.nii.gz` or any file containing "T1" in the name
- Functional file can be named `func.nii.gz`, `rest.nii.gz`, or any `.nii.gz` file in the func/ directory

## Usage

### Quick Start Examples

#### Single Subject (CLI)

Process one subject end-to-end with the default config:

```bash
python pipelines/single_subject.py sub-001 data/raw results
```

With custom configuration and selective steps:

```bash
python pipelines/single_subject.py \
    sub-001 data/raw results \
    --config config/pipeline_config.yaml \
    --skip smoothing
```

#### Multiple Subjects (CLI)

Process every subject found under `data/raw` in parallel:

```bash
python pipelines/batch_processing.py data/raw results
```

Only process specific IDs with limited workers:

```bash
python pipelines/batch_processing.py \
    data/raw results \
    --subjects sub-001 sub-002 sub-010 \
    --n-jobs 4
```

Sequential (one-at-a-time) processing to reduce memory pressure:

```bash
python pipelines/batch_processing.py data/raw results --sequential
```

### Programmatic Usage (Python API)

Each preprocessing script now exposes its main class through an import-friendly wrapper. This enables notebooks or custom workflows to re-use the exact implementations without invoking subprocesses.

```python
from scripts.preprocessing.slice_timing import SliceTimingCorrection
from scripts.preprocessing.coregistration import Coregistration
from scripts.preprocessing.motion_correction import MotionCorrection
from scripts.utils.helpers import load_config

config = load_config("config/pipeline_config.yaml")
motion = MotionCorrection(config)
stc = SliceTimingCorrection(config)
coreg = Coregistration(config)

mc_out = motion.run(
    subject_id="sub-001",
    func_img="data/raw/sub-001/func/rest.nii.gz",
    output_dir="results/preprocessing/sub-001",
)

st_out = stc.run_3dTshift(mc_out, "results/preprocessing/sub-001/sub-001_stc.nii.gz")

coreg.run(
    subject_id="sub-001",
    func_img=st_out,
    anat_img="data/raw/sub-001/anat/T1.nii.gz",
    output_dir="results/preprocessing/sub-001",
)
```

Wrappers are also available for normalization (`SpatialNormalization`) and smoothing (`SpatialSmoothing`), keeping the import path consistent with the CLI modules.

### Custom Pipeline (CLI)

Skip specific steps:

```bash
python pipelines/single_subject.py sub-001 data/raw results --skip motion_correction slice_timing
```

Use custom configuration:

```bash
python pipelines/single_subject.py sub-001 data/raw results --config my_config.yaml
```

### Run Individual Modules

Each preprocessing and analysis step can be run independently:

#### Motion Correction
```bash
python scripts/preprocessing/01_motion_correction.py \
    data/raw/sub-001/func/rest.nii.gz \
    results/preprocessing/sub-001 \
    --subject sub-001
```

#### Functional Connectivity
```bash
python scripts/connectivity/functional_connectivity.py \
    results/preprocessing/sub-001/sub-001_smoothed.nii.gz \
    results/connectivity/sub-001 \
    --subject sub-001 \
    --atlas AAL
```

#### Network Visualization
```bash
python scripts/visualization/glass_brain_network.py \
    results/connectivity/sub-001/sub-001_fc_correlation.npy \
    results/visualization/sub-001 \
    --subject sub-001 \
    --atlas AAL
```

#### Directed Effective Connectivity (arrow plot)
```bash
python scripts/visualization/effective_connectivity_viz.py \
    results/connectivity/sub-001/sub-001_ec_granger.npy \
    results/visualization/sub-001 \
    --subject sub-001 \
    --method granger \
    --top-k 150
```
The matrix is assumed to have shape `[target, source]`; use `--min-weight` to drop weak links.

#### Directed Effective Connectivity (3D connectome; netplotbrain-style)
```bash
python scripts/visualization/effective_connectivity_glass_brain_3d.py \
    results/connectivity/sub-001/sub-001_ec_granger.npy \
    results/visualization/sub-001 \
    --subject sub-001 \
    --method granger \
    --atlas AAL \
    --style netplotbrain \
    --camera-buttons
```
Use `--export-png` to also save a static PNG (requires `plotly` + `kaleido`). If atlas downloads fail, set `--nilearn-data-dir` (or `NILEARN_DATA`) to point to an existing nilearn dataset cache (e.g., `~/nilearn_data`).

### Helper Scripts (exclude sub-007)

- `python scripts/run_sub007_preproc.py` — preprocess-only run for `sub-007` (skips connectivity/ICA/EC).
- `python scripts/run_connectivity_except_sub007.py --preproc-root /data/data2/dataset/fpreproc/results/preprocessing --output-root /data/data2/dataset/fpreproc/results --resume-missing-only` — run connectivity, ICA, EC, and viz for all subjects **except** `sub-007` using already preprocessed data.
- `python scripts/run_all_except_sub007.py --data-dir /data/data2/dataset/proc --output-dir /data/data2/dataset/fpreproc/results` — full pipeline for all subjects except `sub-007` (add `--sequential` to limit memory).

## Configuration

Edit `config/pipeline_config.yaml` to customize:

### Data Parameters
```yaml
data:
  format: "analyze"
  fmri_start_volume: 7  # Skip first 6 volumes
  tr: 2.0               # Repetition time (seconds)
```

### Preprocessing
```yaml
preprocessing:
  brain_extraction:
    method: "deepbet"  # ants_soft, ants_hard, ants_syn, deepbet, bet, nilearn
    strategy: "soft"   # used with ants_* (soft/hard)
    template: "/data/data2/dataset/fpreproc/template/adult/T_template0.nii.gz"
    probability_mask: "/data/data2/dataset/fpreproc/template/adult/T_template0_BrainCerebellumProbabilityMask.nii.gz"
    registration_mask: "/data/data2/dataset/fpreproc/template/adult/T_template0_BrainCerebellumRegistrationMask.nii.gz"
    deepbet:
      path: "/data/data2/dataset/deepbet"
      threshold: 0.5
      n_dilate: 0
      no_gpu: false
      save_tiv: false

  motion_correction:
    reference_volume: "middle"
    cost_function: "normcorr"

  slice_timing:
    enable: true
    slice_order: "interleaved"
    tpattern: null  # Optional AFNI pattern (e.g., "alt+z"); overrides slice_order mapping

  smoothing:
    fwhm: 6  # Full-width half-maximum (mm)

  temporal_filtering:
    highpass: 0.01  # Hz
    lowpass: 0.1    # Hz

registration:
  func_to_anat:
    method: "ants"        # "ants" for SyN, "flirt" for rigid/affine
    cost_function: "bbr"
    dof: 6

  anat_to_standard:
    method: "ants"        # "ants" or "fnirt"
    template: "MNI152_T1_2mm_brain.nii.gz"
    dof: 12
    nonlinear: true
    ants:
      winsorize: [0.005, 0.995]
      histogram_matching: true
      interpolation: "Linear"
      apply_interpolation: "Linear"
      default_value: 0
      stages:
        - name: "rigid"
          transform: "Rigid[0.1]"
          metric: "MI[{fixed},{moving},0.7,32,Regular,0.25]"
          convergence: "[1000x500x250x100,1e-6,10]"
          shrink_factors: "8x4x2x1"
          smoothing_sigmas: "3x2x1x0vox"
        - name: "affine"
          transform: "Affine[0.1]"
          metric: "MI[{fixed},{moving},0.7,32,Regular,0.25]"
          convergence: "[1000x500x250x100,1e-6,10]"
          shrink_factors: "8x4x2x1"
          smoothing_sigmas: "3x2x1x0vox"
        - name: "syn"
          transform: "SyN[0.1,3,0]"
          metric: "CC[{fixed},{moving},1,4]"
          convergence: "[100x70x50x20,1e-6,10]"
          shrink_factors: "8x4x2x1"
          smoothing_sigmas: "3x2x1x0vox"
```
Brain extraction supports ANTs (soft/hard or SyN), deepbet, FSL BET, or nilearn fallback. For deepbet, install the package or set `brain_extraction.deepbet.path` to a local checkout (default `/data/data2/dataset/deepbet`), and use `no_gpu: true` if CUDA is unavailable.

### Atlases
```yaml
atlas:
  parcellations:
    - name: "AAL"
      n_rois: 116
    - name: "Schaefer"
      n_rois: 400
    - name: "Power"
      n_rois: 264

  default: "AAL"
```

### Connectivity Analysis
```yaml
connectivity:
  functional:
    methods:
      - "correlation"
      - "partial_correlation"
      - "tangent"          # requires multiple subjects to estimate a group mean
    group_mode: false      # enable for multi-subject runs when tangent is needed
    threshold: 0.3
    # Valid values (case-insensitive): correlation, partial correlation,
    # tangent, covariance, precision

  ica:
    n_components: 20
    algorithm: "fastica"

  effective:
    methods:
      - "granger"
      - "transfer_entropy"
    max_lag: 5
```
Single-subject runs automatically skip tangent connectivity when `group_mode` is `false`, falling back to correlation.

### Parallel Processing
```yaml
parallel:
  enable: true
  n_jobs: 4  # -1 for all CPUs
```

## Output

### Directory Structure
```
results/
├── preprocessing/
│   └── sub-001/
│       ├── sub-001_moco.nii.gz              # Motion corrected
│       ├── sub-001_stc.nii.gz               # Slice-time corrected
│       ├── sub-001_func_mni.nii.gz          # Normalized to MNI
│       ├── sub-001_smoothed.nii.gz          # Smoothed
│       ├── sub-001_motion_params.txt        # Motion parameters
│       └── qc/                              # Quality control plots
│
├── connectivity/
│   └── sub-001/
│       ├── sub-001_roi_timeseries.npy       # ROI time series
│       ├── sub-001_fc_correlation.npy       # FC matrix
│       ├── sub-001_graph_metrics.json       # Graph metrics
│       ├── sub-001_ica_components.nii.gz    # ICA components
│       ├── sub-001_ec_granger.npy           # Granger causality
│       └── sub-001_ec_transfer_entropy.npy  # Transfer entropy
│
├── visualization/
│   └── sub-001/
│       ├── sub-001_connectome_glass_brain.png    # Glass brain
│       ├── sub-001_connectome_3d.html            # Interactive 3D
│       ├── sub-001_connectivity_matrix.png       # Matrix heatmap
│       ├── sub-001_tsnr_map.png                  # tSNR map
│       ├── sub-001_ec_granger_directed.png       # Directed EC (granger/TE) arrow plot
│       └── ica_components/                       # ICA visualizations
│
└── logs/
    └── sub-001_20250101_120000.log          # Processing log
```

### Results Files

- **NIfTI images** (`.nii.gz`): Brain images
- **NumPy arrays** (`.npy`): Matrices and time series
- **JSON files** (`.json`): Metadata and metrics
- **PNG/HTML files**: Visualizations

## Quality Control

The pipeline generates QC reports for each subject:

1. **Motion QC**: Motion parameters, framewise displacement
2. **Registration QC**: Overlay plots for coregistration and normalization
3. **tSNR Maps**: Temporal signal-to-noise ratio
4. **Carpet Plots**: Time series heatmaps

Review these files in `results/*/qc/` directories.

## Troubleshooting

### FSL/AFNI/ANTs Not Found

If neuroimaging software is not in your PATH:

```bash
# Add to ~/.bashrc or ~/.zshrc
export FSLDIR=/path/to/fsl
export PATH=$FSLDIR/bin:$PATH

export ANTSPATH=/path/to/ants/bin
export PATH=$ANTSPATH:$PATH

export PATH=$PATH:/path/to/afni
```

### Memory Issues

For large datasets or limited memory:

1. Reduce parallel jobs: `--n-jobs 2`
2. Process sequentially: `--sequential`
3. Adjust smoothing FWHM in config

### Missing Atlases

Run the dependency installer:

```bash
python setup/install_dependencies.py
```

Atlases will be automatically downloaded via nilearn.

## Examples for Epilepsy Research

### Identify Seizure Focus Networks

1. **Run FC analysis with multiple atlases**:
```bash
python scripts/connectivity/functional_connectivity.py \
    data.nii.gz output --atlas Schaefer
```

2. **Compute effective connectivity to identify directionality**:
```bash
python scripts/connectivity/effective_connectivity.py \
    timeseries.npy output
```

3. **Visualize hub regions**:
```bash
python scripts/visualization/glass_brain_network.py \
    connectivity_matrix.npy output
```

### Compare Default Mode Network

ICA analysis automatically identifies DMN:

```bash
python scripts/connectivity/ica_analysis.py \
    preprocessed.nii.gz output --n-components 20
```

Check `output/sub-*/sub-*_ica_metadata.json` for DMN component index.

## Citation

If you use this pipeline in your research, please cite:

- **FSL**: Jenkinson et al., NeuroImage 2012
- **AFNI**: Cox, Computers and Biomedical Research 1996
- **ANTs**: Avants et al., NeuroImage 2011
- **Nilearn**: Abraham et al., Frontiers in Neuroinformatics 2014
- **Brain Connectivity Toolbox**: Rubinov & Sporns, NeuroImage 2010

## License

This pipeline is provided for research purposes. Please ensure you have appropriate licenses for FSL, AFNI, ANTs, and other dependencies.

## Support

For issues or questions:
1. Check the logs in `results/logs/`
2. Review QC outputs in `results/*/qc/`
3. Consult individual script help: `python script.py --help`

## Acknowledgments

Developed for epilepsy fMRI research with optimizations for:
- Resting-state network analysis
- Seizure focus localization
- Inter-ictal connectivity patterns
- Default Mode Network alterations

---

**Note**: This pipeline expects pre-installed neuroimaging software (FSL, AFNI, ANTs). Ensure these are properly configured before running the pipeline.
