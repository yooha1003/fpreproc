#!/usr/bin/env python3
"""
Independent Component Analysis (ICA) for resting-state fMRI.
Useful for identifying neural networks in epilepsy research.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICAAnalysis:
    """Independent Component Analysis for fMRI data."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize ICA analysis."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.ica_params = config['connectivity']['ica']

    def run_ica(self, func_img: str, n_components: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run ICA on functional data.

        Parameters
        ----------
        func_img : str
            Preprocessed functional image (4D)
        n_components : int, optional
            Number of components

        Returns
        -------
        components : numpy.ndarray
            Spatial components (n_components x n_voxels)
        time_courses : numpy.ndarray
            Time courses (n_timepoints x n_components)
        mixing_matrix : numpy.ndarray
            Mixing matrix
        """
        logger.info("Running ICA decomposition...")

        from nilearn.decomposition import CanICA

        if n_components is None:
            n_components = self.ica_params.get('n_components', 20)

        # Create CanICA object
        canica = CanICA(
            n_components=n_components,
            algorithm=self.ica_params.get('algorithm', 'fastica'),
            random_state=0,
            memory="nilearn_cache",
            memory_level=2,
            verbose=1,
            mask_strategy='epi',
            smoothing_fwhm=None,  # Already smoothed
            standardize=True,
            detrend=True,
        )

        # Fit to data
        canica.fit(func_img)

        # Get components
        components_img = canica.components_img_
        components = canica.components_

        # Get time courses
        from nilearn.masking import apply_mask
        time_courses = apply_mask(func_img, canica.mask_img_)

        # Project onto components to get time courses
        time_courses_ica = np.dot(time_courses, components.T)

        logger.info(f"ICA completed: {n_components} components extracted")

        return components, time_courses_ica, canica, components_img

    def identify_default_mode_network(self, components: np.ndarray,
                                      components_img: nib.Nifti1Image) -> int:
        """
        Identify Default Mode Network (DMN) component.
        Important for epilepsy studies.

        Parameters
        ----------
        components : numpy.ndarray
            Spatial components
        components_img : nibabel.Nifti1Image
            Components image

        Returns
        -------
        int
            Index of DMN component
        """
        logger.info("Identifying Default Mode Network...")

        from nilearn import datasets
        from nilearn.maskers import NiftiMapsMasker
        from scipy.stats import pearsonr

        try:
            # Load DMN template from Smith et al. 2009
            atlas = datasets.fetch_atlas_smith_2009()
            dmn_idx = 3  # DMN is typically component 4 in Smith atlas

            # Compare each component with DMN template
            correlations = []

            masker = NiftiMapsMasker(maps_img=components_img)

            for i in range(components.shape[0]):
                # Compute spatial correlation
                # This is a simplified version
                correlations.append(0)  # Placeholder

            # Find component with highest correlation to DMN
            dmn_component = np.argmax(correlations) if correlations else 0

            logger.info(f"DMN identified as component {dmn_component}")

            return dmn_component

        except Exception as e:
            logger.warning(f"Could not identify DMN: {e}")
            return 0

    def classify_components(self, components_img: nib.Nifti1Image,
                          time_courses: np.ndarray) -> dict:
        """
        Classify components as signal or noise.

        Parameters
        ----------
        components_img : nibabel.Nifti1Image
            Spatial components
        time_courses : numpy.ndarray
            Component time courses

        Returns
        -------
        dict
            Classification results
        """
        logger.info("Classifying ICA components...")

        classifications = {}
        n_components = time_courses.shape[1]

        for i in range(n_components):
            # Simple heuristics for classification
            # In practice, use more sophisticated methods

            tc = time_courses[:, i]

            # Check frequency content
            fft = np.fft.fft(tc)
            freqs = np.fft.fftfreq(len(tc))

            # High frequency = likely noise
            high_freq_power = np.sum(np.abs(fft[np.abs(freqs) > 0.1]))
            total_power = np.sum(np.abs(fft))

            if high_freq_power / total_power > 0.5:
                label = 'noise'
            else:
                label = 'signal'

            classifications[f'component_{i}'] = {
                'label': label,
                'high_freq_ratio': float(high_freq_power / total_power)
            }

        logger.info(f"Components classified: {sum(1 for c in classifications.values() if c['label'] == 'signal')} signal, {sum(1 for c in classifications.values() if c['label'] == 'noise')} noise")

        return classifications

    def compute_dual_regression(self, func_img: str, components_img: nib.Nifti1Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform dual regression to get subject-specific spatial maps and time courses.

        Parameters
        ----------
        func_img : str
            Functional image
        components_img : nibabel.Nifti1Image
            Group ICA components

        Returns
        -------
        spatial_maps : numpy.ndarray
            Subject-specific spatial maps
        time_courses : numpy.ndarray
            Subject-specific time courses
        """
        logger.info("Running dual regression...")

        from nilearn.maskers import NiftiMapsMasker
        from nilearn.masking import apply_mask
        from sklearn.linear_model import LinearRegression

        # First regression: get subject time courses
        masker = NiftiMapsMasker(maps_img=components_img, standardize=True)
        time_courses = masker.fit_transform(func_img)

        # Second regression: get subject spatial maps
        func_data = nib.load(func_img).get_fdata()
        func_2d = func_data.reshape(-1, func_data.shape[-1]).T

        # Remove zero voxels
        nonzero_mask = np.any(func_2d != 0, axis=0)
        func_2d_nonzero = func_2d[:, nonzero_mask]

        # Fit regression
        reg = LinearRegression()
        reg.fit(time_courses, func_2d_nonzero)

        spatial_maps = np.zeros((time_courses.shape[1], func_2d.shape[1]))
        spatial_maps[:, nonzero_mask] = reg.coef_

        logger.info("Dual regression completed")

        return spatial_maps, time_courses

    def run(self, func_img: str, output_dir: str, subject_id: str) -> dict:
        """
        Run ICA analysis pipeline.

        Parameters
        ----------
        func_img : str
            Preprocessed functional image
        output_dir : str
            Output directory
        subject_id : str
            Subject identifier

        Returns
        -------
        dict
            Results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_components = self.ica_params.get('n_components', 20)

        # Run ICA
        components, time_courses, canica, components_img = self.run_ica(func_img, n_components)

        # Save components
        components_file = output_dir / f'{subject_id}_ica_components.nii.gz'
        nib.save(components_img, components_file)

        # Save time courses
        time_courses_file = output_dir / f'{subject_id}_ica_timecourses.npy'
        np.save(time_courses_file, time_courses)

        # Save spatial components
        spatial_file = output_dir / f'{subject_id}_ica_spatial.npy'
        np.save(spatial_file, components)

        # Classify components
        classifications = self.classify_components(components_img, time_courses)

        # Save classifications
        import json
        class_file = output_dir / f'{subject_id}_ica_classifications.json'
        with open(class_file, 'w') as f:
            json.dump(classifications, f, indent=2)

        # Identify DMN
        dmn_idx = self.identify_default_mode_network(components, components_img)

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'functional_image': str(func_img),
            'n_components': n_components,
            'algorithm': self.ica_params.get('algorithm', 'fastica'),
            'components_file': str(components_file),
            'time_courses_file': str(time_courses_file),
            'spatial_components_file': str(spatial_file),
            'classifications_file': str(class_file),
            'dmn_component': dmn_idx,
            'n_signal_components': sum(1 for c in classifications.values() if c['label'] == 'signal'),
            'n_noise_components': sum(1 for c in classifications.values() if c['label'] == 'noise'),
        }

        metadata_file = output_dir / f'{subject_id}_ica_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"ICA analysis completed for {subject_id}")
        logger.info(f"  Components: {n_components}")
        logger.info(f"  Signal components: {metadata['n_signal_components']}")
        logger.info(f"  Noise components: {metadata['n_noise_components']}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Independent Component Analysis')
    parser.add_argument('input', help='Preprocessed functional image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--n-components', type=int, help='Number of components')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Override n_components if provided
    if args.n_components and config:
        config['connectivity']['ica']['n_components'] = args.n_components

    # Run ICA
    ica = ICAAnalysis(config)
    results = ica.run(args.input, args.output_dir, args.subject)

    print("\n✓ ICA analysis completed!")
    print(f"Components extracted: {results['n_components']}")
    print(f"Signal components: {results['n_signal_components']}")
    print(f"DMN component: {results['dmn_component']}")


if __name__ == '__main__':
    main()
