#!/usr/bin/env python3
"""
Spatial smoothing for fMRI data.
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


class SpatialSmoothing:
    """Spatial smoothing for fMRI data."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize smoothing."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.smooth_params = config['preprocessing']['smoothing']

    def run_fslmaths(self, input_img: str, output_img: str, fwhm: float) -> str:
        """
        Run FSL fslmaths for smoothing.

        Parameters
        ----------
        input_img : str
            Input image
        output_img : str
            Output smoothed image
        fwhm : float
            Full-width half-maximum in mm

        Returns
        -------
        str
            Output image path
        """
        logger.info(f"Running FSL smoothing (FWHM={fwhm}mm)...")

        # Convert FWHM to sigma (sigma = FWHM / 2.355)
        sigma = fwhm / 2.355

        cmd = [
            'fslmaths',
            input_img,
            '-s', str(sigma),
            output_img,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("FSL smoothing completed successfully")
            return output_img

        except FileNotFoundError:
            logger.warning("fslmaths not found, falling back to AFNI")
            return self.run_3dmerge(input_img, output_img, fwhm)

        except subprocess.CalledProcessError as e:
            logger.error(f"fslmaths failed: {e.stderr}")
            raise

    def run_3dmerge(self, input_img: str, output_img: str, fwhm: float) -> str:
        """
        Run AFNI 3dmerge for smoothing.

        Parameters
        ----------
        input_img : str
            Input image
        output_img : str
            Output smoothed image
        fwhm : float
            Full-width half-maximum in mm

        Returns
        -------
        str
            Output image path
        """
        logger.info(f"Running AFNI smoothing (FWHM={fwhm}mm)...")

        cmd = [
            '3dmerge',
            '-1blur_fwhm', str(fwhm),
            '-doall',
            '-prefix', output_img,
            input_img,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("AFNI smoothing completed successfully")
            return output_img

        except FileNotFoundError:
            logger.warning("3dmerge not found, using nilearn fallback")
            return self.run_nilearn_smooth(input_img, output_img, fwhm)

        except subprocess.CalledProcessError as e:
            logger.error(f"3dmerge failed: {e.stderr}")
            raise

    def run_nilearn_smooth(self, input_img: str, output_img: str, fwhm: float) -> str:
        """
        Run nilearn smoothing.

        Parameters
        ----------
        input_img : str
            Input image
        output_img : str
            Output smoothed image
        fwhm : float
            Full-width half-maximum in mm

        Returns
        -------
        str
            Output image path
        """
        logger.info(f"Running nilearn smoothing (FWHM={fwhm}mm)...")

        from nilearn.image import smooth_img

        # Load and smooth
        img = nib.load(input_img)
        smoothed_img = smooth_img(img, fwhm=fwhm)

        # Save
        nib.save(smoothed_img, output_img)

        logger.info("Nilearn smoothing completed successfully")

        return output_img

    def run(self, input_img: str, output_dir: str, subject_id: str) -> dict:
        """
        Run smoothing pipeline.

        Parameters
        ----------
        input_img : str
            Input normalized functional image
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

        fwhm = self.smooth_params.get('fwhm', 6)

        output_img = output_dir / f'{subject_id}_smoothed.nii.gz'

        # Run smoothing
        try:
            output_img = self.run_fslmaths(input_img, str(output_img), fwhm)
        except:
            logger.warning("FSL smoothing failed, trying AFNI...")
            try:
                output_img = self.run_3dmerge(input_img, str(output_img), fwhm)
            except:
                logger.warning("AFNI smoothing failed, using nilearn fallback...")
                output_img = self.run_nilearn_smooth(input_img, str(output_img), fwhm)

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'input_image': str(input_img),
            'output_image': str(output_img),
            'fwhm': fwhm,
        }

        metadata_file = output_dir / f'{subject_id}_smoothing_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Smoothing completed for {subject_id} (FWHM={fwhm}mm)")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Spatial Smoothing')
    parser.add_argument('input', help='Input functional image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--fwhm', type=float, help='FWHM in mm')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Override FWHM if provided
    if args.fwhm and config:
        config['preprocessing']['smoothing']['fwhm'] = args.fwhm

    # Run smoothing
    smooth = SpatialSmoothing(config)
    results = smooth.run(args.input, args.output_dir, args.subject)

    print("\n✓ Smoothing completed successfully!")
    print(f"Output: {results['output_image']}")
    print(f"FWHM: {results['fwhm']}mm")


if __name__ == '__main__':
    main()
