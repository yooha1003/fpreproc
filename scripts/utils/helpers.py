#!/usr/bin/env python3
"""
Helper utilities for the fMRI preprocessing pipeline.
"""

import os
import yaml
import json
import logging
import nibabel as nib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default config.

    Returns
    -------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        default_config = Path(__file__).parent.parent.parent / 'config' / 'pipeline_config.yaml'
        config_path = default_config

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(log_dir: Optional[str] = None,
                  level: str = 'INFO',
                  subject_id: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for pipeline execution.

    Parameters
    ----------
    log_dir : str, optional
        Directory for log files
    level : str
        Logging level
    subject_id : str, optional
        Subject ID for log filename

    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger('fmri_pipeline')
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if subject_id:
            log_file = log_dir / f'{subject_id}_{timestamp}.log'
        else:
            log_file = log_dir / f'pipeline_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Save all details to file
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Logging to: {log_file}")

    return logger


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """
    Save metadata to JSON file.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary
    output_path : str
        Output JSON file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    metadata_converted = {k: convert_types(v) for k, v in metadata.items()}

    with open(output_path, 'w') as f:
        json.dump(metadata_converted, f, indent=2)

    logging.info(f"Metadata saved to: {output_path}")


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load metadata from JSON file.

    Parameters
    ----------
    metadata_path : str
        Path to metadata JSON file

    Returns
    -------
    dict
        Metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def create_output_directory(base_dir: str,
                           subject_id: str,
                           analysis_type: str) -> Path:
    """
    Create standardized output directory structure.

    Parameters
    ----------
    base_dir : str
        Base results directory
    subject_id : str
        Subject identifier
    analysis_type : str
        Type of analysis (e.g., 'preprocessing', 'connectivity')

    Returns
    -------
    pathlib.Path
        Created output directory
    """
    output_dir = Path(base_dir) / analysis_type / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_fsldir() -> Optional[Path]:
    """
    Get FSL installation directory.

    Returns
    -------
    pathlib.Path or None
        FSL directory path
    """
    fsldir = os.environ.get('FSLDIR')

    if fsldir and Path(fsldir).exists():
        return Path(fsldir)
    else:
        logging.warning("FSLDIR not set or directory not found")
        return None


def get_standard_template(template_name: str = 'MNI152_T1_2mm_brain') -> Optional[Path]:
    """
    Get path to standard brain template.

    Parameters
    ----------
    template_name : str
        Template name (without extension)

    Returns
    -------
    pathlib.Path or None
        Template file path
    """
    fsldir = get_fsldir()

    if fsldir:
        template_path = fsldir / 'data' / 'standard' / f'{template_name}.nii.gz'
        if template_path.exists():
            return template_path

    # Try nilearn templates
    try:
        from nilearn import datasets
        if 'MNI152' in template_name:
            mni = datasets.load_mni152_template()
            return Path(mni)
    except:
        pass

    logging.warning(f"Template not found: {template_name}")
    return None


def compute_brain_mask(img: nib.Nifti1Image,
                       threshold: float = 0.5) -> nib.Nifti1Image:
    """
    Compute brain mask from image.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input image
    threshold : float
        Threshold for mask (fraction of robust max)

    Returns
    -------
    nibabel.Nifti1Image
        Binary brain mask
    """
    from nilearn.masking import compute_epi_mask

    mask = compute_epi_mask(img)

    return mask


def extract_time_series(img: nib.Nifti1Image,
                        mask: Union[nib.Nifti1Image, np.ndarray]) -> np.ndarray:
    """
    Extract time series from masked voxels.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        4D fMRI image
    mask : nibabel.Nifti1Image or numpy.ndarray
        3D binary mask

    Returns
    -------
    numpy.ndarray
        Time series (n_timepoints x n_voxels)
    """
    from nilearn.masking import apply_mask

    time_series = apply_mask(img, mask)

    return time_series


def save_nifti(data: np.ndarray,
               affine: np.ndarray,
               output_path: str,
               header: Optional[nib.Nifti1Header] = None) -> None:
    """
    Save data as NIfTI file.

    Parameters
    ----------
    data : numpy.ndarray
        Image data
    affine : numpy.ndarray
        Affine transformation matrix
    output_path : str
        Output file path
    header : nibabel.Nifti1Header, optional
        NIfTI header
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(output_path))

    logging.info(f"Saved: {output_path}")


def plot_registration_overlay(fixed: nib.Nifti1Image,
                              moving: nib.Nifti1Image,
                              output_path: str,
                              title: str = 'Registration Overlay') -> None:
    """
    Create overlay plot for registration QC.

    Parameters
    ----------
    fixed : nibabel.Nifti1Image
        Fixed (reference) image
    moving : nibabel.Nifti1Image
        Moving (registered) image
    output_path : str
        Output file path
    title : str
        Plot title
    """
    from nilearn import plotting

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create overlay plot
    display = plotting.plot_anat(fixed, title=title, display_mode='ortho')
    display.add_edges(moving, color='r')

    # Save
    display.savefig(str(output_path), dpi=150)
    display.close()

    logging.info(f"Registration QC plot saved: {output_path}")


def estimate_framewise_displacement(motion_params: np.ndarray,
                                   radius: float = 50.0) -> np.ndarray:
    """
    Compute framewise displacement from motion parameters.

    Parameters
    ----------
    motion_params : numpy.ndarray
        Motion parameters (n_timepoints x 6): [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
    radius : float
        Head radius in mm (default 50mm)

    Returns
    -------
    numpy.ndarray
        Framewise displacement values
    """
    # Convert rotations to mm (assuming radians)
    motion_mm = motion_params.copy()
    motion_mm[:, 3:] = motion_mm[:, 3:] * radius

    # Calculate displacement
    fd = np.sum(np.abs(np.diff(motion_mm, axis=0)), axis=1)

    # Prepend 0 for first timepoint
    fd = np.concatenate([[0], fd])

    return fd


def create_confound_regressors(motion_params: np.ndarray,
                               include_derivatives: bool = True,
                               include_squared: bool = True) -> np.ndarray:
    """
    Create confound regressors from motion parameters.

    Parameters
    ----------
    motion_params : numpy.ndarray
        Motion parameters (n_timepoints x 6)
    include_derivatives : bool
        Include temporal derivatives
    include_squared : bool
        Include squared parameters

    Returns
    -------
    numpy.ndarray
        Confound regressors
    """
    regressors = [motion_params]

    if include_derivatives:
        derivatives = np.vstack([np.zeros(6), np.diff(motion_params, axis=0)])
        regressors.append(derivatives)

    if include_squared:
        squared = motion_params ** 2
        regressors.append(squared)

        if include_derivatives:
            squared_derivatives = derivatives ** 2
            regressors.append(squared_derivatives)

    confounds = np.hstack(regressors)

    return confounds


if __name__ == '__main__':
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Number of preprocessing steps: {len(config['preprocessing'])}")

    # Test logging setup
    logger = setup_logging(level='DEBUG')
    logger.info("Logging test successful!")
