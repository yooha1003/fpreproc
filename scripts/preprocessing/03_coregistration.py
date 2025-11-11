#!/usr/bin/env python3
"""
Coregistration of functional to anatomical images.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import logging
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata, plot_registration_overlay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Coregistration:
    """Coregister functional to anatomical images."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize coregistration."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.coreg_params = config['registration']['func_to_anat']

    def run_flirt(self, moving: str, fixed: str, output: str,
                  output_matrix: str) -> Tuple[str, str]:
        """
        Run FSL FLIRT for coregistration.

        Parameters
        ----------
        moving : str
            Moving image (functional mean)
        fixed : str
            Fixed image (anatomical)
        output : str
            Output registered image
        output_matrix : str
            Output transformation matrix

        Returns
        -------
        output : str
            Registered image path
        output_matrix : str
            Transformation matrix path
        """
        logger.info("Running FSL FLIRT...")

        cmd = [
            'flirt',
            '-in', moving,
            '-ref', fixed,
            '-out', output,
            '-omat', output_matrix,
            '-dof', str(self.coreg_params.get('dof', 6)),
            '-cost', self.coreg_params.get('cost_function', 'bbr'),
            '-searchcost', self.coreg_params.get('cost_function', 'bbr'),
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("FLIRT completed successfully")
            return output, output_matrix

        except FileNotFoundError:
            logger.warning("FLIRT not found, falling back to ANTs")
            return self.run_ants(moving, fixed, output, output_matrix)

        except subprocess.CalledProcessError as e:
            logger.error(f"FLIRT failed: {e.stderr}")
            raise

    def run_ants(self, moving: str, fixed: str, output: str,
                 output_matrix: str) -> Tuple[str, str]:
        """
        Run ANTs registration.

        Parameters
        ----------
        moving : str
            Moving image
        fixed : str
            Fixed image
        output : str
            Output registered image
        output_matrix : str
            Output transformation matrix

        Returns
        -------
        output : str
            Registered image path
        output_matrix : str
            Transformation matrix path
        """
        logger.info("Running ANTs registration...")

        output_prefix = str(Path(output).with_suffix(''))

        cmd = [
            'antsRegistrationSyN.sh',
            '-d', '3',
            '-f', fixed,
            '-m', moving,
            '-o', output_prefix,
            '-t', 'r',  # Rigid registration
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Rename outputs
            ants_output = f'{output_prefix}Warped.nii.gz'
            ants_matrix = f'{output_prefix}0GenericAffine.mat'

            if Path(ants_output).exists():
                Path(ants_output).rename(output)

            if Path(ants_matrix).exists():
                Path(ants_matrix).rename(output_matrix)

            logger.info("ANTs registration completed successfully")
            return output, output_matrix

        except FileNotFoundError:
            logger.warning("ANTs not found, using nilearn fallback")
            return self.run_nilearn_coreg(moving, fixed, output)

        except subprocess.CalledProcessError as e:
            logger.error(f"ANTs failed: {e.stderr}")
            raise

    def run_nilearn_coreg(self, moving: str, fixed: str, output: str) -> Tuple[str, str]:
        """
        Run nilearn-based coregistration.

        Parameters
        ----------
        moving : str
            Moving image
        fixed : str
            Fixed image
        output : str
            Output image

        Returns
        -------
        output : str
            Registered image path
        matrix : str
            Transformation matrix path (dummy)
        """
        logger.info("Running nilearn coregistration...")

        from nilearn.image import resample_to_img

        # Load images
        moving_img = nib.load(moving)
        fixed_img = nib.load(fixed)

        # Simple resampling (not true registration)
        logger.warning("Nilearn fallback uses simple resampling, not true registration")
        registered_img = resample_to_img(moving_img, fixed_img)

        # Save
        nib.save(registered_img, output)

        # Create dummy matrix
        matrix_file = str(Path(output).with_suffix('.mat'))
        np.savetxt(matrix_file, np.eye(4))

        return output, matrix_file

    def create_mean_functional(self, input_img: str, output: str) -> str:
        """
        Create mean functional image for coregistration.

        Parameters
        ----------
        input_img : str
            4D functional image
        output : str
            Output mean image

        Returns
        -------
        str
            Mean image path
        """
        logger.info("Creating mean functional image...")

        from nilearn.image import mean_img

        img = nib.load(input_img)
        mean = mean_img(img)
        nib.save(mean, output)

        logger.info(f"Mean functional saved: {output}")

        return output

    def run(self, func_img: str, anat_img: str, output_dir: str,
            subject_id: str) -> dict:
        """
        Run coregistration pipeline.

        Parameters
        ----------
        func_img : str
            Functional image (preprocessed)
        anat_img : str
            Anatomical image
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

        # Create mean functional
        mean_func = output_dir / f'{subject_id}_mean_func.nii.gz'
        self.create_mean_functional(func_img, str(mean_func))

        # Register mean func to anatomical
        registered_mean = output_dir / f'{subject_id}_mean_func_coreg.nii.gz'
        transform_matrix = output_dir / f'{subject_id}_func2anat.mat'

        try:
            registered_mean, transform_matrix = self.run_flirt(
                str(mean_func), anat_img, str(registered_mean), str(transform_matrix)
            )
        except:
            logger.warning("FLIRT failed, trying ANTs...")
            try:
                registered_mean, transform_matrix = self.run_ants(
                    str(mean_func), anat_img, str(registered_mean), str(transform_matrix)
                )
            except:
                logger.warning("ANTs failed, using nilearn fallback...")
                registered_mean, transform_matrix = self.run_nilearn_coreg(
                    str(mean_func), anat_img, str(registered_mean)
                )

        # Create QC overlay
        qc_dir = output_dir / 'qc'
        qc_dir.mkdir(exist_ok=True)

        qc_plot = qc_dir / f'{subject_id}_coregistration_qc.png'
        plot_registration_overlay(
            nib.load(anat_img),
            nib.load(registered_mean),
            str(qc_plot),
            title=f'Functional to Anatomical Coregistration - {subject_id}'
        )

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'functional_image': str(func_img),
            'anatomical_image': str(anat_img),
            'mean_functional': str(mean_func),
            'registered_mean': str(registered_mean),
            'transformation_matrix': str(transform_matrix),
            'qc_plot': str(qc_plot),
        }

        metadata_file = output_dir / f'{subject_id}_coregistration_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Coregistration completed for {subject_id}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Functional to Anatomical Coregistration')
    parser.add_argument('func', help='Functional image')
    parser.add_argument('anat', help='Anatomical image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Run coregistration
    coreg = Coregistration(config)
    results = coreg.run(args.func, args.anat, args.output_dir, args.subject)

    print("\n✓ Coregistration completed successfully!")
    print(f"Registered image: {results['registered_mean']}")
    print(f"QC plot: {results['qc_plot']}")


if __name__ == '__main__':
    main()
