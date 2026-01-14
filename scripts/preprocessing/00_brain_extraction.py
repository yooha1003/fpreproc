#!/usr/bin/env python3
"""
Brain extraction (skull-stripping) for anatomical (T1) images.
Uses ANTs `antsBrainExtraction.sh` when available, with a nilearn-based fallback.
"""

import sys
import subprocess
import logging
import os
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (  # noqa: E402
    load_config,
    save_metadata,
    compute_brain_mask,
    get_standard_template,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainExtraction:
    """Perform brain extraction on anatomical images."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.params = config.get('preprocessing', {}).get('brain_extraction', {})
        # 강제: YH + ANTs soft 전략만 사용
        self.method = 'ants_soft'
        self.params['method'] = self.method
        self.params['strategy'] = 'soft'
        base_dir = Path(__file__).resolve().parents[2]
        self.fallback_ants_script = base_dir / 'setup' / 'antsBrainExtraction.sh'
        self.utils_ants_script = base_dir / 'scripts' / 'utils' / 'antsBrainExtraction.sh'
        self.yh_script = base_dir / 'scripts' / 'utils' / 'yhBrainExt.sh'
        self._ants_script = shutil.which('antsBrainExtraction.sh')
        if not self._ants_script:
            if self.fallback_ants_script.exists():
                self._ants_script = str(self.fallback_ants_script)
            elif self.utils_ants_script.exists():
                self._ants_script = str(self.utils_ants_script)

        self._ants_registration = shutil.which('antsRegistration') or shutil.which('antsRegistrationSyN.sh')
        ants_lib = Path('/opt/ants/lib')
        ants_bin = Path('/opt/ants/bin')
        conda_ants_lib = Path('/home/owl/miniconda3/pkgs/ants-2.5.1-h00ab1b0_0/lib')
        self._ants_env = os.environ.copy()
        ld_paths = []
        for candidate in [ants_lib, conda_ants_lib,
                          Path('/home/owl/miniconda3/envs/fmri_pipeline/lib'),
                          Path('/home/owl/miniconda3/lib')]:
            if candidate.exists():
                ld_paths.append(str(candidate))
        if ld_paths:
            current = self._ants_env.get('LD_LIBRARY_PATH', '')
            joined = ":".join(ld_paths)
            path_str = f"{joined}:{current}" if current else joined
            self._ants_env['LD_LIBRARY_PATH'] = path_str
        bin_paths = []
        for candidate in [Path('/home/owl/miniconda3/envs/fmri_pipeline/bin'), ants_bin]:
            if candidate.exists():
                bin_paths.append(str(candidate))
        if bin_paths:
            path_current = self._ants_env.get('PATH', '')
            joined = ":".join(bin_paths)
            path_str = f"{joined}:{path_current}" if path_current else joined
            self._ants_env['PATH'] = path_str
        path_current = self._ants_env.get('PATH', '')
        extra_paths = []
        if self.fallback_ants_script.exists():
            extra_paths.append(str(self.fallback_ants_script.parent))
        if self.utils_ants_script.exists():
            extra_paths.append(str(self.utils_ants_script.parent))
        for candidate in ['/opt/ants/bin', '/home/owl/fsl/bin', '/home/owl/fsl/share/fsl/bin']:
            if Path(candidate).exists():
                extra_paths.append(candidate)
        if extra_paths:
            prepend = ":".join(extra_paths)
            path_str = f"{prepend}:{path_current}" if path_current else prepend
            self._ants_env['PATH'] = path_str
        # Ensure antsBrainExtraction.sh is executable if present
        for script in [self._ants_script, str(self.fallback_ants_script), str(self.utils_ants_script)]:
            if script and Path(script).exists() and not os.access(script, os.X_OK):
                os.chmod(script, 0o755)
        self._fslreorient = shutil.which('fslreorient2std')
        self._bet = shutil.which('bet')
        self.strategy = self.params.get('strategy', 'soft').lower()
        self.keep_intermediate = bool(self.params.get('keep_intermediate', False))

    def _resolve_template_path(self, candidate: Optional[str]) -> Optional[Path]:
        """Resolve template/probability mask paths using FSL/nilearn helpers."""
        if not candidate:
            return None

        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

        resolved = get_standard_template(candidate)
        if resolved:
            return Path(resolved)

        return candidate_path

    def _run_cmd(self, cmd: list, desc: str):
        """Run a shell command with ANTs/FSL environment."""
        logger.info("%s", desc)
        logger.info("Command: %s", ' '.join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=self._ants_env)

    def _reorient_to_std(self, src: str, dst: str) -> str:
        """Reorient image to standard orientation using FSL if available."""
        if self._fslreorient:
            self._run_cmd([self._fslreorient, src, dst], "Reorienting to standard (FSL)")
            return dst
        # fallback: copy
        shutil.copy(src, dst)
        logger.warning("fslreorient2std not found; copied image without reorientation.")
        return dst


    def _run_yh_brainext(self, anat_img: str, output_dir: Path, subject_id: str,
                         strategy: str, template: str, prob_mask: str,
                         reg_mask: Optional[str]) -> Tuple[Path, Path]:
        """
        Run yhBrainExt workflow (via system call to antsBrainExtraction.sh).
        Mirrors the original script behavior as closely as practical.
        """
        if not self._ants_script:
            raise FileNotFoundError("antsBrainExtraction.sh not found in PATH or setup.")

        # Step 1: reorient input to std and strip extension
        reoriented = output_dir / f'{subject_id}_reorient.nii.gz'
        self._reorient_to_std(anat_img, str(reoriented))

        base_in = output_dir / f'{subject_id}_yh'
        shutil.copy(str(reoriented), f'{base_in}.nii.gz')

        # Step 2: N4 bias correction
        bias_img = output_dir / f'{subject_id}_bias.nii.gz'
        self._run_cmd([
            'N4BiasFieldCorrection', '-d', '3', '-i', f'{base_in}.nii.gz',
            '-b', '[200]', '-o', str(bias_img), '-v'
        ], "N4 bias correction")

        # Step 3: antsRegistration init (soft)
        init_prefix = output_dir / 'Tmp'
        self._run_cmd([
            'antsRegistration',
            '--collapse-output-transforms', '1',
            '--dimensionality', '3',
            '--interpolation', 'Linear',
            '--output', f'[{init_prefix},{init_prefix}1Warp.nii.gz,{init_prefix}1WarpInv.nii.gz]',
            '--use-histogram-matching', '1',
            '--winsorize-image-intensities', '[0.005,0.995]',
            '--initial-moving-transform', f'[{template},{bias_img},0]',
            '--transform', 'Rigid[0.1]',
            '--metric', f'MI[{template},{bias_img},1,32,Regular,0.25]',
            '--convergence', '[1000x500,1e-6,10]',
            '--shrink-factors', '12x8',
            '--smoothing-sigmas', '4x3vox'
        ], "antsRegistration init")

        # Copy warp output to *_MNI_init as script does
        init_warp = init_prefix.parent / f'{init_prefix.name}1Warp.nii.gz'
        init_img = output_dir / f'{subject_id}_MNI_init.nii.gz'
        if init_warp.exists():
            shutil.copy(str(init_warp), str(init_img))
        else:
            init_img = bias_img  # fallback

        # Step 4: antsBrainExtraction.sh on init image
        ext_prefix = output_dir / 'Ext'
        cmd_be = [
            self._ants_script,
            '-d', '3',
            '-a', str(init_img if strategy == 'soft' else bias_img),
            '-e', template,
            '-m', prob_mask,
            '-o', str(ext_prefix)
        ]
        if reg_mask:
            cmd_be.extend(['-f', reg_mask])
        self._run_cmd(cmd_be, "antsBrainExtraction.sh")

        # Step 5: back to bias space (soft) or smoothing (hard)
        brain_file = output_dir / f'{subject_id}_anat_brain.nii.gz'
        mask_file = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'

        if strategy == 'soft':
            # Inverse affine only (matching original script use of Tmp0GenericAffine)
            affine_mat = init_prefix.parent / f'{init_prefix.name}0GenericAffine.mat'
            transforms = [f'[{affine_mat},1]'] if affine_mat.exists() else []
            self._apply_transforms_to_mask(
                str(ext_prefix) + "BrainExtractionBrain.nii.gz",
                str(bias_img),
                transforms,
                str(brain_file)
            )
        else:
            # Hard: use BE brain directly with slight erosion (skip smoothing to keep simple)
            be_brain = ext_prefix.parent / f'{ext_prefix.name}BrainExtractionBrain.nii.gz'
            shutil.copy(str(be_brain), str(brain_file))

        # Mask creation from brain > 0
        brain_img = nib.load(str(brain_file))
        mask_data = (brain_img.get_fdata() > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask_data, brain_img.affine, brain_img.header), str(mask_file))

        return brain_file, mask_file

    def _n4_bias_correct(self, input_img: str, output_img: str) -> str:
        """Perform N4 bias field correction."""
        cmd = [
            'N4BiasFieldCorrection',
            '-d', '3',
            '-i', input_img,
            '-b', '[200]',
            '-o', output_img,
            '-v'
        ]
        self._run_cmd(cmd, "N4 bias correction")
        return output_img

    def _ants_registration_init(self, moving_img: str, template: str,
                                output_prefix: Path) -> dict:
        """Initial rigid registration (soft strategy)."""
        cmd = [
            'antsRegistration',
            '--collapse-output-transforms', '1',
            '--dimensionality', '3',
            '--interpolation', 'Linear',
            '--output', f'[{output_prefix},{output_prefix}1Warp.nii.gz,{output_prefix}1WarpInv.nii.gz]',
            '--use-histogram-matching', '1',
            '--winsorize-image-intensities', '[0.005,0.995]',
            '--initial-moving-transform', f'[{template},{moving_img},0]',
            '--transform', 'Rigid[0.1]',
            '--metric', f'MI[{template},{moving_img},1,32,Regular,0.25]',
            '--convergence', '[1000x500,1e-6,10]',
            '--shrink-factors', '12x8',
            '--smoothing-sigmas', '4x3vox'
        ]
        self._run_cmd(cmd, "ANTs initial rigid registration (soft strategy)")
        return {
            'warp': f'{output_prefix}1Warp.nii.gz',
            'inv_warp': f'{output_prefix}1WarpInv.nii.gz',
            'affine': f'{output_prefix}0GenericAffine.mat',
            'warped': f'{output_prefix}Warped.nii.gz'
        }

    def _ants_brain_extraction_script(self, input_img: str, template: str,
                                      prob_mask: str, reg_mask: Optional[str],
                                      output_prefix: Path) -> Path:
        """Run antsBrainExtraction.sh (local copy or system)."""
        script_path = self._ants_script or 'antsBrainExtraction.sh'
        cmd = [
            script_path,
            '-d', '3',
            '-a', input_img,
            '-e', template,
            '-m', prob_mask,
            '-o', str(output_prefix)
        ]
        if reg_mask:
            cmd.extend(['-f', reg_mask])
        self._run_cmd(cmd, "ANTs brain extraction (script)")
        return output_prefix

    def _apply_transforms_to_mask(self, mask_img: str, reference_img: str,
                                  transforms: list, output_img: str):
        """Apply transforms to bring mask into reference space."""
        cmd = [
            'antsApplyTransforms',
            '--dimensionality', '3',
            '--input', mask_img,
            '--reference-image', reference_img,
            '--output', output_img,
            '--n', 'NearestNeighbor'
        ]
        for t in transforms:
            cmd.extend(['--transform', t])
        self._run_cmd(cmd, "Applying transforms to mask")

    def run_ants(self, anat_img: str, template: str, prob_mask: str,
                 output_prefix: Path, reg_mask: Optional[str] = None) -> Tuple[Path, Path]:
        """Run ANTs brain extraction (antsBrainExtraction.sh or fallback script)."""
        if not template or not prob_mask:
            raise FileNotFoundError("Brain extraction template/probability mask not specified.")

        template_path = self._resolve_template_path(template)
        prob_mask_path = self._resolve_template_path(prob_mask)
        reg_mask_path = self._resolve_template_path(reg_mask) if reg_mask else None

        if template_path is None or prob_mask_path is None:
            raise FileNotFoundError("Brain extraction template/probability mask not specified.")

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        if not prob_mask_path.exists():
            raise FileNotFoundError(f"Probability mask not found: {prob_mask_path}")

        self._ants_brain_extraction_script(
            anat_img,
            str(template_path),
            str(prob_mask_path),
            str(reg_mask_path) if reg_mask_path else None,
            output_prefix
        )

        brain_src = output_prefix.parent / f"{output_prefix.name}BrainExtractionBrain.nii.gz"
        mask_src = output_prefix.parent / f"{output_prefix.name}BrainExtractionMask.nii.gz"
        if not brain_src.exists():
            alt_brain = output_prefix.parent / f"{output_prefix.name}Brain.nii.gz"
            if alt_brain.exists():
                brain_src = alt_brain
        if not mask_src.exists():
            alt_mask = output_prefix.parent / f"{output_prefix.name}BrainExtractionMask.nii.gz"
            if alt_mask.exists():
                mask_src = alt_mask

        if not brain_src.exists() or not mask_src.exists():
            raise FileNotFoundError(
                f"Expected ANTs outputs not found: {brain_src}, {mask_src}"
            )

        brain_dest = output_prefix.parent / f"{output_prefix.name}_anat_brain.nii.gz"
        mask_dest = output_prefix.parent / f"{output_prefix.name}_anat_brain_mask.nii.gz"

        brain_src.rename(brain_dest)
        mask_src.rename(mask_dest)

        return brain_dest, mask_dest

    def run_ants_syn(self, anat_img: str, template: str, prob_mask: str,
                     output_dir: Path, subject_id: str) -> Tuple[Path, Path]:
        """
        ANTs-based skull-stripping using antsRegistrationSyN + warped template mask.
        """
        template_path = self._resolve_template_path(template)
        prob_mask_path = self._resolve_template_path(prob_mask)

        if template_path is None or prob_mask_path is None:
            raise FileNotFoundError("Brain extraction template/probability mask not specified.")

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        if not prob_mask_path.exists():
            raise FileNotFoundError(f"Probability mask not found: {prob_mask_path}")

        prefix = output_dir / f"{subject_id}_antsbe_"

        cmd_reg = [
            'antsRegistrationSyN.sh',
            '-d', '3',
            '-f', str(template_path),
            '-m', anat_img,
            '-o', str(prefix)
        ]

        logger.info("Running antsRegistrationSyN-based brain extraction...")
        logger.info("Command: %s", ' '.join(cmd_reg))
        try:
            subprocess.run(cmd_reg, check=True, capture_output=True, text=True, env=self._ants_env)
        except subprocess.CalledProcessError as e:
            logger.error("antsRegistrationSyN failed: %s", e.stderr)
            raise

        mask_out = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'
        cmd_mask = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(prob_mask_path),
            '-r', anat_img,
            '-o', str(mask_out),
            '-n', 'NearestNeighbor',
            '-t', f'{prefix}1InverseWarp.nii.gz',
            '-t', f'[{prefix}0GenericAffine.mat,1]'
        ]

        logger.info("Warping template probability mask to subject space...")
        logger.info("Command: %s", ' '.join(cmd_mask))
        try:
            subprocess.run(cmd_mask, check=True, capture_output=True, text=True, env=self._ants_env)
        except subprocess.CalledProcessError as e:
            logger.error("antsApplyTransforms failed: %s", e.stderr)
            raise

        # Binarize probability mask
        mask_img = nib.load(str(mask_out))
        mask_data = (mask_img.get_fdata() > 0.5).astype(np.uint8)
        mask_bin = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
        nib.save(mask_bin, str(mask_out))

        # Apply mask to anatomy
        anat = nib.load(anat_img)
        brain_data = anat.get_fdata() * mask_data
        brain_img = nib.Nifti1Image(brain_data, anat.affine, anat.header)
        brain_file = output_dir / f'{subject_id}_anat_brain.nii.gz'
        nib.save(brain_img, str(brain_file))

        return brain_file, mask_out

    def run_nilearn_fallback(self, anat_img: str, output_dir: Path,
                             subject_id: str) -> Tuple[Path, Path]:
        """
        Fallback brain extraction using nilearn masking.
        """
        logger.warning("Falling back to nilearn.compute_brain_mask for skull-stripping.")
        anat = nib.load(anat_img)

        mask_img = compute_brain_mask(anat)
        mask_file = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'
        nib.save(mask_img, str(mask_file))

        brain_data = anat.get_fdata() * mask_img.get_fdata()
        brain_img = nib.Nifti1Image(brain_data, anat.affine, anat.header)
        brain_file = output_dir / f'{subject_id}_anat_brain.nii.gz'
        nib.save(brain_img, str(brain_file))

        return brain_file, mask_file

    def run_bet(self, anat_img: str, output_dir: Path, subject_id: str) -> Tuple[Path, Path]:
        """
        FSL BET fallback for skull-stripping.
        """
        brain_file = output_dir / f'{subject_id}_anat_brain.nii.gz'
        mask_file = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'

        frac = str(self.params.get('bet', {}).get('frac', 0.5))
        robust = self.params.get('bet', {}).get('robust', True)
        cmd = ['bet', anat_img, str(brain_file), '-m', '-f', frac]
        if robust:
            cmd.append('-R')

        logger.info("Running FSL BET for brain extraction...")
        logger.info("Command: %s", ' '.join(cmd))

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            logger.warning("BET not found: %s", e)
            raise
        except subprocess.CalledProcessError as e:
            logger.error("BET failed: %s", e.stderr)
            raise

        # BET writes mask as *_mask.nii.gz
        bet_mask = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'
        if not bet_mask.exists():
            bet_mask_default = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'
            if bet_mask_default.exists():
                bet_mask = bet_mask_default
        return brain_file, bet_mask

    def _extract_soft(self, anat_img: str, output_dir: Path, subject_id: str,
                      template: str, prob_mask: str, reg_mask: Optional[str]) -> Tuple[Path, Path]:
        """Soft strategy: reorient -> N4 -> init reg -> antsBrainExtraction on registered image -> bring back."""
        # Reorient -> N4
        reoriented = output_dir / f'{subject_id}_reorient.nii.gz'
        self._reorient_to_std(anat_img, str(reoriented))

        bias_img = output_dir / f'{subject_id}_bias.nii.gz'
        self._n4_bias_correct(str(reoriented), str(bias_img))

        # Initial rigid registration to template
        init_template = self._resolve_template_path(self.params.get('init_template')) or self._resolve_template_path(template)
        if init_template is None or not init_template.exists():
            init_template = self._resolve_template_path(template)

        reg_info = None
        registered_img = str(bias_img)
        if self._ants_registration and init_template and init_template.exists():
            try:
                init_prefix = output_dir / f'{subject_id}_init_'
                reg_info = self._ants_registration_init(str(bias_img), str(init_template), init_prefix)
                warped = Path(reg_info['warped'])
                if warped.exists():
                    registered_img = str(warped)
            except Exception as e:
                logger.warning("Initial registration failed (%s); continuing without init reg.", e)

        # antsBrainExtraction on registered image
        ext_prefix = output_dir / f'{subject_id}_ext_'
        brain_be, mask_be = self.run_ants(
            registered_img,
            template,
            prob_mask,
            ext_prefix,
            reg_mask=reg_mask
        )

        # Bring mask back to bias space if registration was run
        mask_native = mask_be
        if reg_info and Path(reg_info['inv_warp']).exists() and Path(reg_info['affine']).exists():
            mask_native = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'
            transforms = [
                reg_info['inv_warp'],
                f"[{reg_info['affine']},1]"
            ]
            self._apply_transforms_to_mask(str(mask_be), str(bias_img), transforms, str(mask_native))

        # Apply mask to bias image
        mask_img = nib.load(str(mask_native))
        mask_data = (mask_img.get_fdata() > 0.5).astype(np.uint8)
        bias = nib.load(str(bias_img))
        brain_data = bias.get_fdata() * mask_data
        brain_file = output_dir / f'{subject_id}_anat_brain.nii.gz'
        nib.save(nib.Nifti1Image(brain_data, bias.affine, bias.header), str(brain_file))
        nib.save(nib.Nifti1Image(mask_data, bias.affine, bias.header), str(mask_native))

        # Optional BET refine similar to reference script
        if self._bet:
            try:
                bet_out = output_dir / f'{subject_id}_anat_brain_bet.nii.gz'
                cmd = [
                    self._bet,
                    str(brain_file),
                    str(bet_out),
                    '-f', '0.4',
                    '-R'
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True, env=self._ants_env)
                # Use BET output if created
                if bet_out.exists():
                    brain_file = bet_out
                    mask_candidate = output_dir / f'{subject_id}_anat_brain_bet_mask.nii.gz'
                    if mask_candidate.exists():
                        mask_native = mask_candidate
            except Exception as e:
                logger.warning("BET refinement failed (%s); using original mask.", e)

        return brain_file, Path(mask_native)

    def _extract_hard(self, anat_img: str, output_dir: Path, subject_id: str,
                      template: str, prob_mask: str, reg_mask: Optional[str]) -> Tuple[Path, Path]:
        """Hard strategy: reorient -> N4 -> direct antsBrainExtraction on bias image."""
        reoriented = output_dir / f'{subject_id}_reorient.nii.gz'
        self._reorient_to_std(anat_img, str(reoriented))
        bias_img = output_dir / f'{subject_id}_bias.nii.gz'
        self._n4_bias_correct(str(reoriented), str(bias_img))

        ext_prefix = output_dir / f'{subject_id}_ext_'
        brain_be, mask_be = self.run_ants(
            str(bias_img),
            template,
            prob_mask,
            ext_prefix,
            reg_mask=reg_mask
        )

        # Optionally smooth/erode mask based on voxel size
        mask_native = mask_be
        try:
            img_mask = nib.load(str(mask_be))
            resol = nib.load(str(bias_img)).header.get_zooms()[0]
            from scipy.ndimage import gaussian_filter

            smoothed = gaussian_filter(img_mask.get_fdata(), sigma=resol)
            mask_data = (smoothed > 0.5).astype(np.uint8)
            mask_native = output_dir / f'{subject_id}_anat_brain_mask.nii.gz'
            nib.save(nib.Nifti1Image(mask_data, img_mask.affine, img_mask.header), str(mask_native))
        except Exception as e:
            logger.warning("Mask smoothing skipped (%s)", e)

        # Apply mask
        mask_img = nib.load(str(mask_native))
        bias = nib.load(str(bias_img))
        brain_data = bias.get_fdata() * (mask_img.get_fdata() > 0.5)
        brain_file = output_dir / f'{subject_id}_anat_brain.nii.gz'
        nib.save(nib.Nifti1Image(brain_data, bias.affine, bias.header), str(brain_file))
        return brain_file, Path(mask_native)

    def run(self, anat_img: str, output_dir: str, subject_id: str) -> dict:
        """
        Execute brain extraction workflow.

        Returns metadata including output file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        brain_file: Path
        mask_file: Path
        method_used = None

        try:
            template = self.params.get('template')
            prob_mask = self.params.get('probability_mask')
            reg_mask = self.params.get('registration_mask')

            brain_file, mask_file = self._run_yh_brainext(
                anat_img, output_dir, subject_id, self.strategy, template, prob_mask, reg_mask
            )
            method_used = f'ants_{self.strategy}'
        except Exception as e:
            logger.warning("ANTs brain extraction failed (%s); attempting BET fallback.", e)
            try:
                brain_file, mask_file = self.run_bet(anat_img, output_dir, subject_id)
                method_used = 'bet'
            except Exception as e2:
                logger.warning("BET failed (%s); using nilearn fallback.", e2)
                brain_file, mask_file = self.run_nilearn_fallback(anat_img, output_dir, subject_id)
                method_used = 'nilearn'

        metadata = {
            'subject_id': subject_id,
            'input_anatomical': str(anat_img),
            'brain_image': str(brain_file),
            'brain_mask': str(mask_file),
            'method': method_used,
            'strategy': self.strategy,
        }

        # QC overlay: background (yh/reorient/original) + extracted brain (combined sagittal/coronal/axial)
        try:
            qc_dir = output_dir / 'qc'
            qc_dir.mkdir(exist_ok=True)
            qc_file = qc_dir / f'{subject_id}_brain_extraction_qc.png'
            import matplotlib.pyplot as plt
            from nilearn import plotting

            bg_candidates = [
                output_dir / f'{subject_id}_yh.nii.gz',
                output_dir / f'{subject_id}_reorient.nii.gz',
                Path(anat_img)
            ]
            bg_img = next((p for p in bg_candidates if Path(p).exists()), Path(anat_img))

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='none')
            views = ['x', 'y', 'z']
            cuts = [[0], [0], [0]]  # center slice per view

            for ax, mode, cut in zip(axes, views, cuts):
                disp = plotting.plot_anat(
                    str(bg_img),
                    display_mode=mode,
                    cut_coords=cut,
                    draw_cross=False,
                    axes=ax,
                    annotate=False
                )
                disp.add_overlay(str(brain_file), cmap='hot', alpha=0.5)
                ax.set_title("")
                ax.axis("off")
                ax.set_frame_on(False)
                ax.set_facecolor('none')

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
            fig.savefig(str(qc_file), bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            metadata['qc_overlay'] = str(qc_file)
        except Exception as e:
            logger.warning("QC overlay generation failed: %s", e)

        metadata_file = output_dir / f'{subject_id}_brain_extraction_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info("Brain extraction completed for %s", subject_id)
        logger.info("Brain: %s", brain_file)
        logger.info("Mask: %s", mask_file)

        return metadata


def main():
    """CLI entry point for brain extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Brain extraction for anatomical images')
    parser.add_argument('anat', help='Input anatomical image (T1)')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    config = load_config(args.config) if args.config else None

    extractor = BrainExtraction(config)
    results = extractor.run(args.anat, args.output_dir, args.subject)

    print("\n✓ Brain extraction completed")
    print(f"Brain image: {results['brain_image']}")
    print(f"Mask: {results['brain_mask']}")


if __name__ == '__main__':
    main()
