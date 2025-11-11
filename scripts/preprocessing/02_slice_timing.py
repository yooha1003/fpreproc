#!/usr/bin/env python3
"""
Slice timing correction for fMRI data.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import logging
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SliceTimingCorrection:
    """Slice timing correction for fMRI data."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize slice timing correction.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.st_params = config['preprocessing']['slice_timing']
        self.tr = config['data']['tr']

    def run_3dTshift(self, input_img: str, output_img: str) -> str:
        """
        Run AFNI 3dTshift for slice timing correction.

        Parameters
        ----------
        input_img : str
            Input motion-corrected fMRI image
        output_img : str
            Output slice-time corrected image

        Returns
        -------
        str
            Output image path
        """
        logger.info("Running AFNI 3dTshift...")

        # Build command
        cmd = [
            '3dTshift',
            '-prefix', output_img,
            '-TR', str(self.tr),
        ]

        # Slice order
        slice_order = self.st_params.get('slice_order', 'interleaved')
        if slice_order == 'interleaved':
            cmd.append('-altplus')  # Odd slices first (1,3,5..., then 2,4,6...)
        elif slice_order == 'sequential_ascending':
            cmd.append('-sequential')
        elif slice_order == 'sequential_descending':
            cmd.append('-seqminus')

        cmd.append(input_img)

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("3dTshift completed successfully")
            return output_img

        except FileNotFoundError:
            logger.warning("3dTshift not found, falling back to nilearn")
            return self.run_nilearn_stc(input_img, output_img)

        except subprocess.CalledProcessError as e:
            logger.error(f"3dTshift failed: {e.stderr}")
            raise

    def run_nilearn_stc(self, input_img: str, output_img: str) -> str:
        """
        Run nilearn-based slice timing correction.

        Parameters
        ----------
        input_img : str
            Input image path
        output_img : str
            Output image path

        Returns
        -------
        str
            Output image path
        """
        logger.info("Running nilearn slice timing correction...")

        from nilearn.image import index_img
        from scipy.interpolate import interp1d

        # Load image
        img = nib.load(input_img)
        data = img.get_fdata()
        n_slices = data.shape[2]
        n_timepoints = data.shape[3]

        # Get slice acquisition times
        slice_order = self.st_params.get('slice_order', 'interleaved')

        if slice_order == 'interleaved':
            # Odd slices first
            slice_indices = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
        elif slice_order == 'sequential_ascending':
            slice_indices = list(range(n_slices))
        elif slice_order == 'sequential_descending':
            slice_indices = list(range(n_slices-1, -1, -1))
        else:
            slice_indices = list(range(n_slices))

        # Acquisition times for each slice
        slice_times = np.zeros(n_slices)
        for i, idx in enumerate(slice_indices):
            slice_times[idx] = i * (self.tr / n_slices)

        # Reference time (middle slice)
        ref_time = self.tr / 2

        # Perform slice timing correction
        corrected_data = np.zeros_like(data)

        for z in range(n_slices):
            # Original time points for this slice
            orig_times = np.arange(n_timepoints) * self.tr + slice_times[z]

            # Target time points (aligned to reference)
            target_times = np.arange(n_timepoints) * self.tr + ref_time

            # Interpolate each voxel in this slice
            for x in range(data.shape[0]):
                for y in range(data.shape[1]):
                    time_series = data[x, y, z, :]

                    # Linear interpolation
                    f = interp1d(orig_times, time_series, kind='linear',
                               bounds_error=False, fill_value='extrapolate')

                    corrected_data[x, y, z, :] = f(target_times)

        # Save corrected image
        corrected_img = nib.Nifti1Image(corrected_data, img.affine, img.header)
        nib.save(corrected_img, output_img)

        logger.info(f"Slice timing correction completed (nilearn)")

        return output_img

    def run(self, input_img: str, output_dir: str, subject_id: str) -> dict:
        """
        Run slice timing correction pipeline.

        Parameters
        ----------
        input_img : str
            Input motion-corrected fMRI image
        output_dir : str
            Output directory
        subject_id : str
            Subject identifier

        Returns
        -------
        dict
            Results including paths
        """
        if not self.st_params.get('enable', True):
            logger.info("Slice timing correction disabled, skipping...")
            return {
                'subject_id': subject_id,
                'input_image': str(input_img),
                'output_image': str(input_img),
                'status': 'skipped'
            }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_img = output_dir / f'{subject_id}_stc.nii.gz'

        # Run slice timing correction
        try:
            output_img = self.run_3dTshift(str(input_img), str(output_img))
        except:
            logger.warning("3dTshift failed, using nilearn fallback...")
            output_img = self.run_nilearn_stc(str(input_img), str(output_img))

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'input_image': str(input_img),
            'output_image': str(output_img),
            'tr': self.tr,
            'slice_order': self.st_params.get('slice_order', 'interleaved'),
        }

        metadata_file = output_dir / f'{subject_id}_slice_timing_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Slice timing correction completed for {subject_id}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='fMRI Slice Timing Correction')
    parser.add_argument('input', help='Input motion-corrected fMRI image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--tr', type=float, help='Repetition time (TR) in seconds')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Override TR if provided
    if args.tr and config:
        config['data']['tr'] = args.tr

    # Run slice timing correction
    stc = SliceTimingCorrection(config)
    results = stc.run(args.input, args.output_dir, args.subject)

    if results['status'] != 'skipped':
        print("\n✓ Slice timing correction completed successfully!")
        print(f"Output: {results['output_image']}")


if __name__ == '__main__':
    main()
