# fMRI Preprocessing and Network Analysis Pipeline

A comprehensive, modular pipeline for preprocessing resting-state fMRI data and performing connectivity analysis, with a focus on epilepsy research.

## Features

### Preprocessing
- **Motion Correction**: FSL MCFLIRT / AFNI 3dvolreg
- **Slice Timing Correction**: AFNI 3dTshift
- **Spatial Registration**: Functional to anatomical (FLIRT/ANTs)
- **Normalization**: MNI standard space (FNIRT/ANTs SyN)
- **Spatial Smoothing**: Gaussian smoothing (FSL/AFNI)

### Connectivity Analysis
- **Functional Connectivity (FC)**
  - Pearson correlation
  - Partial correlation
  - Tangent space embedding
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

### Pipeline Execution
- **Single Subject**: Process one subject end-to-end
- **Batch Processing**: Parallel or sequential processing of multiple subjects
- **Modular Design**: Skip or customize any pipeline step

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
│   │   └── activation_patterns.py
│   │
│   └── utils/                        # Utility functions
│       ├── data_loader.py
│       ├── helpers.py
│       └── quality_control.py
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

### Option 1: 3D Series (Multiple Files per Subject)
```
data/raw/
├── sub-001/
│   ├── anat/
│   │   ├── T1.hdr
│   │   └── T1.img
│   └── func/
│       ├── xxxx_001_004_ep2d_fid_basic_bold_p2_Epilepsy-1_0007.hdr
│       ├── xxxx_001_004_ep2d_fid_basic_bold_p2_Epilepsy-1_0007.img
│       ├── xxxx_001_004_ep2d_fid_basic_bold_p2_Epilepsy-1_0008.hdr
│       ├── xxxx_001_004_ep2d_fid_basic_bold_p2_Epilepsy-1_0008.img
│       └── ... (multiple 3D volumes from 0007 to 0281+)
├── sub-002/
│   └── ...
└── sub-003/
    └── ...
```

### Option 2: Single 4D File per Subject
```
data/raw/
├── sub-001/
│   ├── anat/
│   │   ├── T1.hdr
│   │   └── T1.img
│   └── func/
│       ├── rest.hdr
│       └── rest.img
├── sub-002/
│   └── ...
```

**Important Notes**:
- The pipeline automatically detects and handles both formats
- For **3D series**: Files are sorted by volume number (e.g., `_0007`, `_0008`) and concatenated into 4D
- The **first 6 volumes** are automatically removed (configurable: `fmri_start_volume: 7` in `config/pipeline_config.yaml`)
- Files must be in **Analyze format** (`.hdr`/`.img` pairs)

## Usage

### Quick Start: Single Subject

Process one subject through the entire pipeline:

```bash
python pipelines/single_subject.py sub-001 data/raw results
```

### Batch Processing

Process all subjects in parallel:

```bash
python pipelines/batch_processing.py data/raw results
```

Process specific subjects:

```bash
python pipelines/batch_processing.py data/raw results --subjects sub-001 sub-002 sub-003
```

Process sequentially (not in parallel):

```bash
python pipelines/batch_processing.py data/raw results --sequential
```

Limit parallel jobs:

```bash
python pipelines/batch_processing.py data/raw results --n-jobs 4
```

### Custom Pipeline

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
  motion_correction:
    reference_volume: "middle"
    cost_function: "normcorr"

  slice_timing:
    enable: true
    slice_order: "interleaved"

  smoothing:
    fwhm: 6  # Full-width half-maximum (mm)

  temporal_filtering:
    highpass: 0.01  # Hz
    lowpass: 0.1    # Hz
```

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
      - "tangent"
    threshold: 0.3

  ica:
    n_components: 20
    algorithm: "fastica"

  effective:
    methods:
      - "granger"
      - "transfer_entropy"
    max_lag: 5
```

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
