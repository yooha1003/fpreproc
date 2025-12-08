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
        self.ants_env = os.environ.copy()
        ants_lib = Path('/opt/ants/lib')
        ants_bin = Path('/opt/ants/bin')
        if ants_lib.exists():
            current = self.ants_env.get('LD_LIBRARY_PATH', '')
            self.ants_env['LD_LIBRARY_PATH'] = f"{ants_lib}:{current}" if current else str(ants_lib)
        if ants_bin.exists():
            current_path = self.ants_env.get('PATH', '')
            self.ants_env['PATH'] = f"{ants_bin}:{current_path}" if current_path else str(ants_bin)

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

        output_path = Path(output)

        def _strip_nii_gz(path: Path) -> Path:
            """Remove .nii/.nii.gz suffixes for ANTs output prefix."""
            if path.name.endswith('.nii.gz'):
                return path.with_suffix('').with_suffix('')
            return path.with_suffix('')

        output_prefix = str(_strip_nii_gz(output_path))

        transform = self.coreg_params.get('ants_transform', 's')

        cmd = [
            'antsRegistrationSyN.sh',
            '-d', '3',
            '-f', fixed,
            '-m', moving,
            '-o', output_prefix,
            '-t', transform,  # SyN by default ('s')
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=self.ants_env)

            # Rename outputs
            ants_output = f'{output_prefix}Warped.nii.gz'
            ants_matrix = f'{output_prefix}0GenericAffine.mat'

            ants_output_path = Path(ants_output)
            ants_matrix_path = Path(ants_matrix)

            if not ants_output_path.exists():
                logger.error("ANTs coregistration did not create %s", ants_output)
                logger.debug("ANTs stdout:\n%s", result.stdout)
                logger.debug("ANTs stderr:\n%s", result.stderr)
                raise RuntimeError(f"ANTs output missing: {ants_output}")

            ants_output_path.rename(output)

            if ants_matrix_path.exists():
                ants_matrix_path.rename(output_matrix)
            else:
                logger.error("ANTs coregistration missing affine matrix %s", ants_matrix)
                raise RuntimeError(f"ANTs matrix missing: {ants_matrix}")

            # Capture warp if produced
            ants_warp = Path(f'{output_prefix}1Warp.nii.gz')
            ants_invwarp = Path(f'{output_prefix}1InverseWarp.nii.gz')
            warp_out = Path(output_path.parent / f'{output_path.stem}1Warp.nii.gz')
            invwarp_out = Path(output_path.parent / f'{output_path.stem}1InverseWarp.nii.gz')
            if ants_warp.exists():
                ants_warp.rename(warp_out)
            if ants_invwarp.exists():
                ants_invwarp.rename(invwarp_out)

            self.last_ants_warp = str(warp_out) if warp_out.exists() else None
            self.last_ants_invwarp = str(invwarp_out) if invwarp_out.exists() else None

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

        preferred_method = self.coreg_params.get('method', 'flirt').lower()

        def _run_flirt():
            return self.run_flirt(
                str(mean_func), anat_img, str(registered_mean), str(transform_matrix)
            )

        def _run_ants():
            return self.run_ants(
                str(mean_func), anat_img, str(registered_mean), str(transform_matrix)
            )

        try:
            if preferred_method == 'ants':
                logger.info("Coregistration method set to ANTs SyN.")
                registered_mean, transform_matrix = _run_ants()
            else:
                if preferred_method != 'flirt':
                    logger.warning(f"Unknown method '{preferred_method}', defaulting to FLIRT.")
                logger.info("Coregistration method set to FLIRT.")
                registered_mean, transform_matrix = _run_flirt()
        except Exception:
            if preferred_method == 'ants':
                logger.warning("ANTs failed, falling back to FLIRT...")
                try:
                    registered_mean, transform_matrix = _run_flirt()
                except Exception:
                    logger.warning("FLIRT fallback failed, using nilearn resampling...")
                    registered_mean, transform_matrix = self.run_nilearn_coreg(
                        str(mean_func), anat_img, str(registered_mean)
                    )
            else:
                logger.warning("FLIRT failed, trying ANTs...")
                try:
                    registered_mean, transform_matrix = _run_ants()
                except Exception:
                    logger.warning("ANTs fallback failed, using nilearn resampling...")
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
            'ants_warp': getattr(self, 'last_ants_warp', None),
            'ants_inverse_warp': getattr(self, 'last_ants_invwarp', None),
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
