#!/usr/bin/env python3
"""
Single subject fMRI preprocessing and analysis pipeline.
Runs complete analysis for one subject from raw data to visualization.
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from utils.helpers import load_config, setup_logging, save_metadata
from utils.data_loader import AnalyzeDataLoader
from utils.quality_control import QualityControl

# Preprocessing
from preprocessing.motion_correction import MotionCorrection
from preprocessing.slice_timing import SliceTimingCorrection
from preprocessing.coregistration import Coregistration
from preprocessing.normalization import SpatialNormalization
from preprocessing.smoothing import SpatialSmoothing

# Connectivity
from connectivity.functional_connectivity import FunctionalConnectivity
from connectivity.ica_analysis import ICAAnalysis
from connectivity.effective_connectivity import EffectiveConnectivity

# Visualization
from visualization.glass_brain_network import GlassBrainNetworkViz
from visualization.activation_patterns import ActivationPatternViz


class SingleSubjectPipeline:
    """Complete single subject processing pipeline."""

    def __init__(self, config_path: str = None):
        """
        Initialize pipeline.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = None

    def run(self, subject_id: str, data_dir: str, output_dir: str,
            skip_steps: list = None) -> dict:
        """
        Run complete pipeline for single subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., 'sub-001')
        data_dir : str
            Data directory containing raw data
        output_dir : str
            Output directory for results
        skip_steps : list, optional
            List of steps to skip

        Returns
        -------
        dict
            Pipeline results and metadata
        """
        if skip_steps is None:
            skip_steps = []

        # Setup logging
        log_dir = Path(output_dir) / 'logs'
        self.logger = setup_logging(
            log_dir=str(log_dir),
            level=self.config['logging']['level'],
            subject_id=subject_id
        )

        self.logger.info("="*80)
        self.logger.info(f"Starting single subject pipeline: {subject_id}")
        self.logger.info("="*80)

        start_time = datetime.now()

        # Create output directories
        output_dir = Path(output_dir)
        preproc_dir = output_dir / 'preprocessing' / subject_id
        conn_dir = output_dir / 'connectivity' / subject_id
        viz_dir = output_dir / 'visualization' / subject_id

        for d in [preproc_dir, conn_dir, viz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        results = {
            'subject_id': subject_id,
            'start_time': str(start_time),
            'steps_completed': [],
            'steps_skipped': skip_steps,
        }

        try:
            # =================================================================
            # STEP 1: Load Data
            # =================================================================
            if 'load_data' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 1: Loading Data")
                self.logger.info("="*80)

                loader = AnalyzeDataLoader(data_dir)

                # Validate subject data
                validation = loader.validate_subject_data(subject_id)
                if not validation['valid']:
                    self.logger.error(f"Data validation failed: {validation['errors']}")
                    raise ValueError(f"Invalid subject data: {validation['errors']}")

                # Load functional data
                fmri_start_volume = self.config['data'].get('fmri_start_volume', 7)
                func_img, func_meta = loader.load_fmri_data(subject_id, fmri_start_volume)

                # Save loaded functional data
                func_file = preproc_dir / f'{subject_id}_func_raw.nii.gz'
                import nibabel as nib
                nib.save(func_img, func_file)

                # Load anatomical data
                anat_img, anat_meta = loader.load_anatomical_data(subject_id)

                # Save loaded anatomical data
                anat_file = preproc_dir / f'{subject_id}_anat_raw.nii.gz'
                nib.save(anat_img, anat_file)

                self.logger.info(f"Loaded functional data: {func_meta['remaining_volumes']} volumes")
                self.logger.info(f"Loaded anatomical data: {anat_meta['shape']}")

                results['raw_functional'] = str(func_file)
                results['raw_anatomical'] = str(anat_file)
                results['steps_completed'].append('load_data')

            # =================================================================
            # STEP 2: Motion Correction
            # =================================================================
            if 'motion_correction' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 2: Motion Correction")
                self.logger.info("="*80)

                mc = MotionCorrection(self.config)
                mc_results = mc.run(
                    str(func_file),
                    str(preproc_dir),
                    subject_id
                )

                func_file = mc_results['output_image']
                results['motion_correction'] = mc_results
                results['steps_completed'].append('motion_correction')

            # =================================================================
            # STEP 3: Slice Timing Correction
            # =================================================================
            if 'slice_timing' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 3: Slice Timing Correction")
                self.logger.info("="*80)

                stc = SliceTimingCorrection(self.config)
                stc_results = stc.run(
                    str(func_file),
                    str(preproc_dir),
                    subject_id
                )

                if stc_results['status'] != 'skipped':
                    func_file = stc_results['output_image']

                results['slice_timing'] = stc_results
                results['steps_completed'].append('slice_timing')

            # =================================================================
            # STEP 4: Coregistration
            # =================================================================
            if 'coregistration' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 4: Coregistration (Functional to Anatomical)")
                self.logger.info("="*80)

                coreg = Coregistration(self.config)
                coreg_results = coreg.run(
                    str(func_file),
                    str(anat_file),
                    str(preproc_dir),
                    subject_id
                )

                results['coregistration'] = coreg_results
                results['steps_completed'].append('coregistration')

            # =================================================================
            # STEP 5: Spatial Normalization
            # =================================================================
            if 'normalization' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 5: Spatial Normalization to MNI")
                self.logger.info("="*80)

                norm = SpatialNormalization(self.config)
                norm_results = norm.run(
                    str(anat_file),
                    str(func_file),
                    str(preproc_dir),
                    subject_id
                )

                func_file = norm_results['normalized_functional']
                results['normalization'] = norm_results
                results['steps_completed'].append('normalization')

            # =================================================================
            # STEP 6: Spatial Smoothing
            # =================================================================
            if 'smoothing' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 6: Spatial Smoothing")
                self.logger.info("="*80)

                smooth = SpatialSmoothing(self.config)
                smooth_results = smooth.run(
                    str(func_file),
                    str(preproc_dir),
                    subject_id
                )

                func_file = smooth_results['output_image']
                results['smoothing'] = smooth_results
                results['preprocessed_functional'] = str(func_file)
                results['steps_completed'].append('smoothing')

            # =================================================================
            # STEP 7: Functional Connectivity
            # =================================================================
            if 'functional_connectivity' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 7: Functional Connectivity Analysis")
                self.logger.info("="*80)

                fc = FunctionalConnectivity(self.config)
                fc_results = fc.run(
                    str(func_file),
                    str(conn_dir),
                    subject_id
                )

                results['functional_connectivity'] = fc_results
                results['steps_completed'].append('functional_connectivity')

            # =================================================================
            # STEP 8: ICA Analysis
            # =================================================================
            if 'ica' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 8: Independent Component Analysis")
                self.logger.info("="*80)

                ica = ICAAnalysis(self.config)
                ica_results = ica.run(
                    str(func_file),
                    str(conn_dir),
                    subject_id
                )

                results['ica'] = ica_results
                results['steps_completed'].append('ica')

            # =================================================================
            # STEP 9: Effective Connectivity
            # =================================================================
            if 'effective_connectivity' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 9: Effective Connectivity Analysis")
                self.logger.info("="*80)

                # Use time series from FC analysis
                ts_file = fc_results['time_series_file']

                ec = EffectiveConnectivity(self.config)
                ec_results = ec.run(
                    str(ts_file),
                    str(conn_dir),
                    subject_id
                )

                results['effective_connectivity'] = ec_results
                results['steps_completed'].append('effective_connectivity')

            # =================================================================
            # STEP 10: Network Visualization
            # =================================================================
            if 'network_viz' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 10: Network Visualization")
                self.logger.info("="*80)

                # Use correlation matrix from FC
                conn_matrix_file = fc_results['connectivity_matrices']['correlation']

                viz_net = GlassBrainNetworkViz(self.config)
                viz_net_results = viz_net.run(
                    str(conn_matrix_file),
                    str(viz_dir),
                    subject_id
                )

                results['network_visualization'] = viz_net_results
                results['steps_completed'].append('network_viz')

            # =================================================================
            # STEP 11: Activation Pattern Visualization
            # =================================================================
            if 'activation_viz' not in skip_steps:
                self.logger.info("\n" + "="*80)
                self.logger.info("STEP 11: Activation Pattern Visualization")
                self.logger.info("="*80)

                viz_act = ActivationPatternViz(self.config)
                viz_act_results = viz_act.run(
                    str(func_file),
                    str(viz_dir),
                    subject_id,
                    ica_components=ica_results.get('components_file')
                )

                results['activation_visualization'] = viz_act_results
                results['steps_completed'].append('activation_viz')

            # =================================================================
            # Finalize
            # =================================================================
            end_time = datetime.now()
            duration = end_time - start_time

            results['end_time'] = str(end_time)
            results['duration_seconds'] = duration.total_seconds()
            results['status'] = 'completed'

            self.logger.info("\n" + "="*80)
            self.logger.info(f"Pipeline completed successfully for {subject_id}")
            self.logger.info(f"Duration: {duration}")
            self.logger.info(f"Steps completed: {len(results['steps_completed'])}")
            self.logger.info("="*80)

            # Save results metadata
            results_file = output_dir / f'{subject_id}_pipeline_results.json'
            save_metadata(results, str(results_file))

            self.logger.info(f"\nResults saved to: {results_file}")

            return results

        except Exception as e:
            self.logger.error(f"\n{'='*80}")
            self.logger.error(f"Pipeline failed for {subject_id}")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"{'='*80}")

            results['status'] = 'failed'
            results['error'] = str(e)

            import traceback
            self.logger.error(traceback.format_exc())

            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Single Subject fMRI Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python single_subject.py sub-001 /path/to/data /path/to/output

  # Skip specific steps
  python single_subject.py sub-001 /path/to/data /path/to/output --skip motion_correction

  # Use custom config
  python single_subject.py sub-001 /path/to/data /path/to/output --config my_config.yaml
        """
    )

    parser.add_argument('subject_id', help='Subject identifier (e.g., sub-001)')
    parser.add_argument('data_dir', help='Data directory containing raw data')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--skip', nargs='+', help='Steps to skip')

    args = parser.parse_args()

    # Run pipeline
    pipeline = SingleSubjectPipeline(args.config)

    try:
        results = pipeline.run(
            args.subject_id,
            args.data_dir,
            args.output_dir,
            skip_steps=args.skip
        )

        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Subject: {results['subject_id']}")
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Steps completed: {len(results['steps_completed'])}")
        print(f"\nResults directory: {args.output_dir}")
        print("="*80)

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print("✗ PIPELINE FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print("="*80)

        return 1


if __name__ == '__main__':
    sys.exit(main())
