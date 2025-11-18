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
import shutil
from typing import Optional, Tuple, Union, Dict, Any

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
        self.method = self.norm_params.get('method', 'fnirt').lower()
        self.ants_params = self.norm_params.get('ants', {})

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
                return self.run_ants_registration(anat, template, output)

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

    def run_ants_registration(self, moving: str, fixed: str,
                              output: str) -> Tuple[str, Dict[str, str]]:
        """
        Run ANTs registration using a rigid+affine+SyN workflow.

        Parameters
        ----------
        moving : str
            Moving image (subject anatomical)
        fixed : str
            Fixed reference (MNI template)
        output : str
            Output normalized anatomical image

        Returns
        -------
        Tuple[str, Dict[str, str]]
            Normalized image path and dictionary describing transforms.
        """
        logger.info("Running customized ANTs registration (Rigid+Affine+SyN)...")

        def build_prefix(path: str) -> Path:
            path_obj = Path(path)
            name = path_obj.name
            if name.endswith('.nii.gz'):
                name = name[:-7]
            else:
                name = path_obj.stem
            return path_obj.with_name(name)

        output_prefix_path = build_prefix(output)
        output_prefix = str(output_prefix_path)

        winsorize = self.ants_params.get('winsorize', [0.005, 0.995])
        hist_match = '1' if self.ants_params.get('histogram_matching', True) else '0'
        interpolation = self.ants_params.get('interpolation', 'Linear')
        dimensionality = str(self.ants_params.get('dimensionality', 3))
        float_precision = '1' if self.ants_params.get('float', False) else '0'

        default_stages = [
            {
                'transform': 'Rigid[0.1]',
                'metric': 'MI[{fixed},{moving},0.7,32,Regular,0.25]',
                'convergence': '[1000x500x250x100,1e-6,10]',
                'shrink_factors': '8x4x2x1',
                'smoothing_sigmas': '3x2x1x0vox'
            },
            {
                'transform': 'Affine[0.1]',
                'metric': 'MI[{fixed},{moving},0.7,32,Regular,0.25]',
                'convergence': '[1000x500x250x100,1e-6,10]',
                'shrink_factors': '8x4x2x1',
                'smoothing_sigmas': '3x2x1x0vox'
            },
            {
                'transform': 'SyN[0.1,3,0]',
                'metric': 'CC[{fixed},{moving},1,4]',
                'convergence': '[100x70x50x20,1e-6,10]',
                'shrink_factors': '8x4x2x1',
                'smoothing_sigmas': '3x2x1x0vox'
            }
        ]

        stages = self.ants_params.get('stages', default_stages)

        cmd = [
            'antsRegistration',
            '--dimensionality', dimensionality,
            '--float', float_precision,
            '--output', output_prefix,
            '--interpolation', interpolation,
            '--winsorize-image-intensities',
            f'[{winsorize[0]},{winsorize[1]}]',
            '--use-histogram-matching', hist_match,
            '--initial-moving-transform',
            f'[{fixed},{moving},1]'
        ]

        def replace_placeholders(value: str) -> str:
            return value.replace('{fixed}', fixed).replace('{moving}', moving)

        for stage in stages:
            transform = stage.get('transform')
            if not transform:
                continue
            cmd.extend(['--transform', transform])

            metric = stage.get('metric')
            if metric:
                cmd.extend(['--metric', replace_placeholders(metric)])

            convergence = stage.get('convergence')
            if convergence:
                cmd.extend(['--convergence', convergence])

            shrink = stage.get('shrink_factors')
            if shrink:
                cmd.extend(['--shrink-factors', shrink])

            sigmas = stage.get('smoothing_sigmas')
            if sigmas:
                cmd.extend(['--smoothing-sigmas', sigmas])

        if self.ants_params.get('verbose', True):
            cmd.append('-v')

        logger.info(f"ANTs command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            logger.warning("ANTs not found, using nilearn fallback")
            return self.run_nilearn_normalize(moving, fixed, output)
        except subprocess.CalledProcessError as e:
            logger.error(f"ANTs failed: {e.stderr}")
            raise

        # Expected ANTs outputs based on prefix
        ants_warp = Path(f'{output_prefix}1Warp.nii.gz')
        ants_affine = Path(f'{output_prefix}0GenericAffine.mat')

        # Apply transforms to anatomical image to create final normalized volume
        apply_cmd = [
            'antsApplyTransforms',
            '-d', dimensionality,
            '-i', moving,
            '-r', fixed,
            '-o', output,
            '-n', self.ants_params.get('apply_interpolation', interpolation),
            '--default-value', str(self.ants_params.get('default_value', 0))
        ]

        # Apply warp then affine (last transform listed is applied first)
        if ants_warp.exists():
            apply_cmd.extend(['-t', str(ants_warp)])
        if ants_affine.exists():
            apply_cmd.extend(['-t', str(ants_affine)])

        logger.info(f"Applying ANTs transforms: {' '.join(apply_cmd)}")
        try:
            subprocess.run(apply_cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"antsApplyTransforms failed: {getattr(e, 'stderr', e)}")
            raise

        transforms = {'type': 'ants'}
        if ants_warp.exists():
            warp_field = Path(output).parent / f'{Path(output).stem}_antsWarp.nii.gz'
            shutil.move(str(ants_warp), warp_field)
            transforms['warp'] = str(warp_field)

        if ants_affine.exists():
            affine_file = Path(output).parent / f'{Path(output).stem}_antsAffine.mat'
            shutil.move(str(ants_affine), affine_file)
            transforms['affine'] = str(affine_file)

        # Remove intermediate warped output if it exists (antsRegistration creates prefixWarped.nii.gz)
        warped_tmp = Path(f'{output_prefix}Warped.nii.gz')
        if warped_tmp.exists():
            warped_tmp.unlink()

        logger.info("ANTs registration completed successfully")
        return output, transforms

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

    def apply_normalization_to_functional(self, func_img: str,
                                          transform: Union[str, Dict[str, Any]],
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

        # Handle ANTs transforms explicitly
        if isinstance(transform, dict) and transform.get('type') == 'ants':
            warp = transform.get('warp')
            affine = transform.get('affine')
            cmd = [
                'antsApplyTransforms',
                '-d', '3',
                '-i', func_img,
                '-r', template,
                '-o', output,
            ]

            # Order matters: warp first, then affine
            if warp:
                cmd.extend(['-t', warp])
            if affine:
                cmd.extend(['-t', affine])

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("Applied ANTs transforms to functional data")
                return output
            except (FileNotFoundError, subprocess.CalledProcessError):
                logger.warning("ANTs apply failed, falling back to nilearn resampling")
                from nilearn.image import resample_to_img

                func = nib.load(func_img)
                template_img = nib.load(template)
                normalized = resample_to_img(func, template_img)
                nib.save(normalized, output)
                return output

        warp_field = transform

        # Check if warp is a matrix or warp field
        is_matrix = isinstance(warp_field, str) and warp_field.endswith('.mat')

        if is_matrix and isinstance(warp_field, str):
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

        elif isinstance(warp_field, str):
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

        # Fallback: use ANTs if available (single transform or dict already handled)
        if isinstance(warp_field, str):
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
                pass

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
            if self.method == 'ants':
                normalized_anat, warp_field = self.run_ants_registration(
                    anat_img, str(template), str(normalized_anat)
                )
            else:
                normalized_anat, warp_field = self.run_fnirt(
                    anat_img, str(template), str(normalized_anat), str(warp_field)
                )
        except Exception:
            if self.method == 'ants':
                logger.warning("ANTs failed, falling back to FNIRT...")
                try:
                    normalized_anat, warp_field = self.run_fnirt(
                        anat_img, str(template), str(normalized_anat), str(warp_field)
                    )
                except Exception:
                    logger.warning("FNIRT fallback failed, using nilearn...")
                    normalized_anat, warp_field = self.run_nilearn_normalize(
                        anat_img, str(template), str(normalized_anat)
                    )
            else:
                logger.warning("FNIRT failed, trying ANTs...")
                try:
                    normalized_anat, warp_field = self.run_ants_registration(
                        anat_img, str(template), str(normalized_anat)
                    )
                except Exception:
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
            'warp_field': warp_field,
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
