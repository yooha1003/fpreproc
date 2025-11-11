#!/usr/bin/env python3
"""
Spatial normalization to MNI standard space.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import logging
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata, get_standard_template, plot_registration_overlay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialNormalization:
    """Normalize images to MNI standard space."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize normalization."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.norm_params = config['registration']['anat_to_standard']

    def run_fnirt(self, anat: str, template: str, output: str,
                  warp_field: str) -> Tuple[str, str]:
        """
        Run FSL FNIRT for nonlinear registration.

        Parameters
        ----------
        anat : str
            Anatomical image
        template : str
            Template image (MNI)
        output : str
            Output normalized image
        warp_field : str
            Output warp field

        Returns
        -------
        output : str
            Normalized image path
        warp_field : str
            Warp field path
        """
        logger.info("Running FSL FNIRT (nonlinear registration)...")

        # First run affine registration with FLIRT
        affine_matrix = str(Path(output).with_suffix('.mat'))

        cmd_flirt = [
            'flirt',
            '-in', anat,
            '-ref', template,
            '-omat', affine_matrix,
            '-dof', str(self.norm_params.get('dof', 12)),
            '-cost', self.norm_params.get('cost_function', 'corratio'),
        ]

        logger.info(f"FLIRT command: {' '.join(cmd_flirt)}")

        try:
            subprocess.run(cmd_flirt, check=True, capture_output=True, text=True)
            logger.info("FLIRT affine registration completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"FLIRT failed: {e.stderr}")
            raise

        # Then run nonlinear registration if enabled
        if self.norm_params.get('nonlinear', True):
            cmd_fnirt = [
                'fnirt',
                '--in=' + anat,
                '--ref=' + template,
                '--aff=' + affine_matrix,
                '--iout=' + output,
                '--cout=' + warp_field,
            ]

            logger.info(f"FNIRT command: {' '.join(cmd_fnirt)}")

            try:
                subprocess.run(cmd_fnirt, check=True, capture_output=True, text=True)
                logger.info("FNIRT nonlinear registration completed")
                return output, warp_field

            except subprocess.CalledProcessError as e:
                logger.error(f"FNIRT failed: {e.stderr}")
                raise

            except FileNotFoundError:
                logger.warning("FNIRT not found, falling back to ANTs")
                return self.run_ants_syn(anat, template, output)

        else:
            # Apply affine only
            cmd_apply = [
                'flirt',
                '-in', anat,
                '-ref', template,
                '-out', output,
                '-init', affine_matrix,
                '-applyxfm',
            ]

            subprocess.run(cmd_apply, check=True, capture_output=True, text=True)

            return output, affine_matrix

    def run_ants_syn(self, moving: str, fixed: str, output: str) -> Tuple[str, str]:
        """
        Run ANTs SyN for nonlinear registration.

        Parameters
        ----------
        moving : str
            Moving image (anatomical)
        fixed : str
            Fixed image (template)
        output : str
            Output normalized image

        Returns
        -------
        output : str
            Normalized image path
        warp_field : str
            Warp field path
        """
        logger.info("Running ANTs SyN (nonlinear registration)...")

        output_prefix = str(Path(output).with_suffix(''))

        cmd = [
            'antsRegistrationSyN.sh',
            '-d', '3',
            '-f', fixed,
            '-m', moving,
            '-o', output_prefix,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Rename outputs
            ants_output = f'{output_prefix}Warped.nii.gz'
            ants_warp = f'{output_prefix}1Warp.nii.gz'

            if Path(ants_output).exists():
                Path(ants_output).rename(output)

            if Path(ants_warp).exists():
                warp_field = str(Path(output).parent / f'{Path(output).stem}_warp.nii.gz')
                Path(ants_warp).rename(warp_field)
            else:
                warp_field = f'{output_prefix}0GenericAffine.mat'

            logger.info("ANTs SyN completed successfully")
            return output, warp_field

        except FileNotFoundError:
            logger.warning("ANTs not found, using nilearn fallback")
            return self.run_nilearn_normalize(moving, fixed, output)

        except subprocess.CalledProcessError as e:
            logger.error(f"ANTs failed: {e.stderr}")
            raise

    def run_nilearn_normalize(self, moving: str, fixed: str, output: str) -> Tuple[str, str]:
        """
        Run nilearn-based normalization (simple resampling).

        Parameters
        ----------
        moving : str
            Moving image
        fixed : str
            Fixed image (template)
        output : str
            Output image

        Returns
        -------
        output : str
            Normalized image path
        matrix : str
            Dummy transformation matrix
        """
        logger.info("Running nilearn normalization (simplified)...")
        logger.warning("This is a simplified version using resampling only")

        from nilearn.image import resample_to_img

        moving_img = nib.load(moving)
        fixed_img = nib.load(fixed)

        # Resample to template space
        normalized_img = resample_to_img(moving_img, fixed_img)

        # Save
        nib.save(normalized_img, output)

        # Dummy transformation
        matrix_file = str(Path(output).with_suffix('.mat'))
        np.savetxt(matrix_file, np.eye(4))

        return output, matrix_file

    def apply_normalization_to_functional(self, func_img: str, warp_field: str,
                                         template: str, output: str) -> str:
        """
        Apply normalization transformation to functional data.

        Parameters
        ----------
        func_img : str
            Functional image
        warp_field : str
            Warp field or transformation matrix
        template : str
            Template image
        output : str
            Output normalized functional image

        Returns
        -------
        str
            Normalized functional image path
        """
        logger.info("Applying normalization to functional data...")

        # Check if warp is a matrix or warp field
        is_matrix = warp_field.endswith('.mat')

        if is_matrix:
            # Use FLIRT to apply affine transformation
            try:
                cmd = [
                    'flirt',
                    '-in', func_img,
                    '-ref', template,
                    '-out', output,
                    '-init', warp_field,
                    '-applyxfm',
                ]

                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("Applied affine transformation to functional data")
                return output

            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

        else:
            # Use FNIRT/applywarp for nonlinear transformation
            try:
                cmd = [
                    'applywarp',
                    '--in=' + func_img,
                    '--ref=' + template,
                    '--warp=' + warp_field,
                    '--out=' + output,
                ]

                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("Applied warp field to functional data")
                return output

            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

        # Fallback: use ANTs
        try:
            cmd = [
                'antsApplyTransforms',
                '-d', '3',
                '-i', func_img,
                '-r', template,
                '-t', warp_field,
                '-o', output,
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Applied transformation using ANTs")
            return output

        except (FileNotFoundError, subprocess.CalledProcessError):
            # Final fallback: simple resampling
            logger.warning("Using nilearn resampling fallback")
            from nilearn.image import resample_to_img

            func = nib.load(func_img)
            template_img = nib.load(template)

            normalized = resample_to_img(func, template_img)
            nib.save(normalized, output)

            return output

    def run(self, anat_img: str, func_img: str, output_dir: str,
            subject_id: str) -> dict:
        """
        Run normalization pipeline.

        Parameters
        ----------
        anat_img : str
            Anatomical image
        func_img : str
            Functional image
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

        # Get template
        template_name = self.norm_params.get('template', 'MNI152_T1_2mm_brain')
        template = get_standard_template(template_name)

        if template is None:
            raise FileNotFoundError(f"Template not found: {template_name}")

        logger.info(f"Using template: {template}")

        # Normalize anatomical to MNI
        normalized_anat = output_dir / f'{subject_id}_T1_mni.nii.gz'
        warp_field = output_dir / f'{subject_id}_anat2mni_warp.nii.gz'

        try:
            normalized_anat, warp_field = self.run_fnirt(
                anat_img, str(template), str(normalized_anat), str(warp_field)
            )
        except:
            logger.warning("FNIRT failed, trying ANTs...")
            try:
                normalized_anat, warp_field = self.run_ants_syn(
                    anat_img, str(template), str(normalized_anat)
                )
            except:
                logger.warning("ANTs failed, using nilearn fallback...")
                normalized_anat, warp_field = self.run_nilearn_normalize(
                    anat_img, str(template), str(normalized_anat)
                )

        # Apply to functional data
        normalized_func = output_dir / f'{subject_id}_func_mni.nii.gz'
        self.apply_normalization_to_functional(
            func_img, warp_field, str(template), str(normalized_func)
        )

        # Create QC overlays
        qc_dir = output_dir / 'qc'
        qc_dir.mkdir(exist_ok=True)

        qc_anat = qc_dir / f'{subject_id}_normalization_anat_qc.png'
        plot_registration_overlay(
            nib.load(str(template)),
            nib.load(str(normalized_anat)),
            str(qc_anat),
            title=f'Anatomical Normalization to MNI - {subject_id}'
        )

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'anatomical_image': str(anat_img),
            'functional_image': str(func_img),
            'template': str(template),
            'normalized_anatomical': str(normalized_anat),
            'normalized_functional': str(normalized_func),
            'warp_field': str(warp_field),
            'qc_plot': str(qc_anat),
        }

        metadata_file = output_dir / f'{subject_id}_normalization_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Normalization completed for {subject_id}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Spatial Normalization to MNI')
    parser.add_argument('anat', help='Anatomical image')
    parser.add_argument('func', help='Functional image')
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

    # Run normalization
    norm = SpatialNormalization(config)
    results = norm.run(args.anat, args.func, args.output_dir, args.subject)

    print("\n✓ Normalization completed successfully!")
    print(f"Normalized anatomical: {results['normalized_anatomical']}")
    print(f"Normalized functional: {results['normalized_functional']}")


if __name__ == '__main__':
    main()
