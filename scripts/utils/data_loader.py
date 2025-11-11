#!/usr/bin/env python3
"""
Data loading utilities for Analyze format (hdr/img) fMRI data.
Handles loading and basic validation of neuroimaging data.
"""

import os
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzeDataLoader:
    """Loader for Analyze format neuroimaging data (hdr/img files)."""

    def __init__(self, data_dir: str):
        """
        Initialize data loader.

        Parameters
        ----------
        data_dir : str
            Base directory containing subject data
        """
        self.data_dir = Path(data_dir)

    def load_analyze_image(self, hdr_path: str) -> nib.analyze.AnalyzeImage:
        """
        Load an Analyze format image.

        Parameters
        ----------
        hdr_path : str
            Path to .hdr file

        Returns
        -------
        nibabel.analyze.AnalyzeImage
            Loaded image
        """
        hdr_path = Path(hdr_path)

        if not hdr_path.exists():
            raise FileNotFoundError(f"Header file not found: {hdr_path}")

        # Check for corresponding .img file
        img_path = hdr_path.with_suffix('.img')
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        try:
            img = nib.load(str(hdr_path))
            logger.info(f"Loaded: {hdr_path.name}")
            logger.info(f"  Shape: {img.shape}")
            logger.info(f"  Affine:\n{img.affine}")
            return img

        except Exception as e:
            logger.error(f"Error loading {hdr_path}: {e}")
            raise

    def load_fmri_data(self, subject_id: str, start_volume: int = 7) -> Tuple[nib.Nifti1Image, dict]:
        """
        Load fMRI data for a subject, removing initial volumes.

        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., 'sub-001')
        start_volume : int
            First volume to keep (1-indexed, default=7)

        Returns
        -------
        img : nibabel.Nifti1Image
            Loaded and trimmed fMRI data
        metadata : dict
            Metadata about the loaded data
        """
        func_dir = self.data_dir / subject_id / 'func'

        if not func_dir.exists():
            raise FileNotFoundError(f"Functional directory not found: {func_dir}")

        # Find .hdr file
        hdr_files = list(func_dir.glob('*.hdr'))

        if not hdr_files:
            raise FileNotFoundError(f"No .hdr files found in {func_dir}")

        if len(hdr_files) > 1:
            logger.warning(f"Multiple .hdr files found in {func_dir}. Using first one.")

        hdr_path = hdr_files[0]

        # Load image
        img = self.load_analyze_image(hdr_path)

        # Get data array
        data = img.get_fdata()

        # Check if 4D
        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D data, got shape {data.shape}")

        n_volumes = data.shape[3]
        logger.info(f"Total volumes: {n_volumes}")

        # Remove initial volumes (volumes 1-6, keeping 7 onwards)
        if start_volume > 1:
            logger.info(f"Removing first {start_volume - 1} volumes")
            data_trimmed = data[:, :, :, start_volume - 1:]
            logger.info(f"Remaining volumes: {data_trimmed.shape[3]}")
        else:
            data_trimmed = data

        # Create new NIfTI image
        img_trimmed = nib.Nifti1Image(data_trimmed, img.affine, img.header)

        metadata = {
            'subject_id': subject_id,
            'original_path': str(hdr_path),
            'original_volumes': n_volumes,
            'start_volume': start_volume,
            'remaining_volumes': data_trimmed.shape[3],
            'shape': data_trimmed.shape,
            'voxel_size': img.header.get_zooms()[:3],
        }

        return img_trimmed, metadata

    def load_anatomical_data(self, subject_id: str) -> Tuple[nib.Nifti1Image, dict]:
        """
        Load T1-weighted anatomical data for a subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier

        Returns
        -------
        img : nibabel.Nifti1Image
            Loaded T1 image
        metadata : dict
            Metadata about the loaded data
        """
        anat_dir = self.data_dir / subject_id / 'anat'

        if not anat_dir.exists():
            raise FileNotFoundError(f"Anatomical directory not found: {anat_dir}")

        # Find T1 .hdr file
        hdr_files = list(anat_dir.glob('*T1*.hdr')) or list(anat_dir.glob('*.hdr'))

        if not hdr_files:
            raise FileNotFoundError(f"No .hdr files found in {anat_dir}")

        if len(hdr_files) > 1:
            logger.warning(f"Multiple .hdr files found in {anat_dir}. Using first one.")

        hdr_path = hdr_files[0]

        # Load image
        img = self.load_analyze_image(hdr_path)

        # Convert to NIfTI
        data = img.get_fdata()
        img_nifti = nib.Nifti1Image(data, img.affine, img.header)

        metadata = {
            'subject_id': subject_id,
            'original_path': str(hdr_path),
            'shape': data.shape,
            'voxel_size': img.header.get_zooms()[:3],
        }

        return img_nifti, metadata

    def get_subject_list(self) -> List[str]:
        """
        Get list of all subjects in data directory.

        Returns
        -------
        list of str
            Subject identifiers
        """
        subjects = []

        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                subjects.append(item.name)

        subjects.sort()
        logger.info(f"Found {len(subjects)} subjects: {subjects}")

        return subjects

    def validate_subject_data(self, subject_id: str) -> dict:
        """
        Validate that required data exists for a subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier

        Returns
        -------
        dict
            Validation results
        """
        results = {
            'subject_id': subject_id,
            'valid': True,
            'errors': [],
            'warnings': [],
        }

        subject_dir = self.data_dir / subject_id

        # Check subject directory
        if not subject_dir.exists():
            results['valid'] = False
            results['errors'].append(f"Subject directory not found: {subject_dir}")
            return results

        # Check anatomical data
        anat_dir = subject_dir / 'anat'
        if not anat_dir.exists():
            results['valid'] = False
            results['errors'].append(f"Anatomical directory not found: {anat_dir}")
        else:
            anat_files = list(anat_dir.glob('*.hdr'))
            if not anat_files:
                results['valid'] = False
                results['errors'].append(f"No anatomical .hdr files found in {anat_dir}")

        # Check functional data
        func_dir = subject_dir / 'func'
        if not func_dir.exists():
            results['valid'] = False
            results['errors'].append(f"Functional directory not found: {func_dir}")
        else:
            func_files = list(func_dir.glob('*.hdr'))
            if not func_files:
                results['valid'] = False
                results['errors'].append(f"No functional .hdr files found in {func_dir}")

        return results


def convert_analyze_to_nifti(analyze_path: str, output_path: str) -> None:
    """
    Convert Analyze format to NIfTI format.

    Parameters
    ----------
    analyze_path : str
        Path to .hdr file
    output_path : str
        Output .nii or .nii.gz path
    """
    img = nib.load(analyze_path)
    data = img.get_fdata()

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(data, img.affine, img.header)

    # Save
    nib.save(nifti_img, output_path)
    logger.info(f"Converted {analyze_path} -> {output_path}")


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <data_directory>")
        sys.exit(1)

    data_dir = sys.argv[1]
    loader = AnalyzeDataLoader(data_dir)

    # Get subject list
    subjects = loader.get_subject_list()

    # Validate each subject
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Validating: {subject}")
        print('='*60)

        validation = loader.validate_subject_data(subject)

        if validation['valid']:
            print("✓ Valid")

            # Try loading data
            try:
                fmri_img, fmri_meta = loader.load_fmri_data(subject)
                print(f"  fMRI: {fmri_meta['remaining_volumes']} volumes")

                anat_img, anat_meta = loader.load_anatomical_data(subject)
                print(f"  T1: {anat_meta['shape']}")

            except Exception as e:
                print(f"  ✗ Error loading: {e}")
        else:
            print("✗ Invalid")
            for error in validation['errors']:
                print(f"  - {error}")
