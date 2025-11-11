#!/usr/bin/env python3
"""
Batch processing for multiple subjects.
Processes all subjects in parallel or sequentially.
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from utils.helpers import load_config, setup_logging, save_metadata
from utils.data_loader import NiftiDataLoader

# Import single subject pipeline
from single_subject import SingleSubjectPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processing for multiple subjects."""

    def __init__(self, config_path: str = None):
        """
        Initialize batch processor.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        self.config = load_config(config_path)
        self.config_path = config_path

    def process_subject(self, subject_id: str, data_dir: str,
                       output_dir: str, skip_steps: list = None) -> dict:
        """
        Process a single subject (wrapper for parallel execution).

        Parameters
        ----------
        subject_id : str
            Subject identifier
        data_dir : str
            Data directory
        output_dir : str
            Output directory
        skip_steps : list, optional
            Steps to skip

        Returns
        -------
        dict
            Processing results
        """
        logger.info(f"\nProcessing {subject_id}...")

        try:
            pipeline = SingleSubjectPipeline(self.config_path)
            results = pipeline.run(subject_id, data_dir, output_dir, skip_steps)
            results['success'] = True

            logger.info(f"✓ {subject_id} completed successfully")

            return results

        except Exception as e:
            logger.error(f"✗ {subject_id} failed: {str(e)}")

            return {
                'subject_id': subject_id,
                'success': False,
                'error': str(e),
                'status': 'failed'
            }

    def run_sequential(self, subjects: list, data_dir: str, output_dir: str,
                      skip_steps: list = None) -> dict:
        """
        Run batch processing sequentially.

        Parameters
        ----------
        subjects : list
            List of subject IDs
        data_dir : str
            Data directory
        output_dir : str
            Output directory
        skip_steps : list, optional
            Steps to skip

        Returns
        -------
        dict
            Batch processing results
        """
        logger.info("="*80)
        logger.info("BATCH PROCESSING (Sequential)")
        logger.info("="*80)
        logger.info(f"Subjects: {len(subjects)}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*80)

        start_time = datetime.now()
        results = []

        for i, subject_id in enumerate(subjects, 1):
            logger.info(f"\n[{i}/{len(subjects)}] Processing {subject_id}...")

            result = self.process_subject(subject_id, data_dir, output_dir, skip_steps)
            results.append(result)

        end_time = datetime.now()
        duration = end_time - start_time

        # Summary
        summary = self._create_summary(results, start_time, end_time)

        return summary

    def run_parallel(self, subjects: list, data_dir: str, output_dir: str,
                    skip_steps: list = None, n_jobs: int = None) -> dict:
        """
        Run batch processing in parallel.

        Parameters
        ----------
        subjects : list
            List of subject IDs
        data_dir : str
            Data directory
        output_dir : str
            Output directory
        skip_steps : list, optional
            Steps to skip
        n_jobs : int, optional
            Number of parallel jobs (default: number of CPUs)

        Returns
        -------
        dict
            Batch processing results
        """
        if n_jobs is None:
            n_jobs = self.config['parallel'].get('n_jobs', mp.cpu_count())

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        logger.info("="*80)
        logger.info("BATCH PROCESSING (Parallel)")
        logger.info("="*80)
        logger.info(f"Subjects: {len(subjects)}")
        logger.info(f"Parallel jobs: {n_jobs}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*80)

        start_time = datetime.now()
        results = []

        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_subject = {
                executor.submit(
                    self.process_subject,
                    subject_id, data_dir, output_dir, skip_steps
                ): subject_id
                for subject_id in subjects
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_subject), 1):
                subject_id = future_to_subject[future]

                try:
                    result = future.result()
                    results.append(result)

                    if result['success']:
                        logger.info(f"[{i}/{len(subjects)}] ✓ {subject_id} completed")
                    else:
                        logger.error(f"[{i}/{len(subjects)}] ✗ {subject_id} failed")

                except Exception as e:
                    logger.error(f"[{i}/{len(subjects)}] ✗ {subject_id} failed with exception: {e}")
                    results.append({
                        'subject_id': subject_id,
                        'success': False,
                        'error': str(e),
                        'status': 'failed'
                    })

        end_time = datetime.now()

        # Summary
        summary = self._create_summary(results, start_time, end_time)

        return summary

    def _create_summary(self, results: list, start_time: datetime,
                       end_time: datetime) -> dict:
        """
        Create summary of batch processing results.

        Parameters
        ----------
        results : list
            List of individual subject results
        start_time : datetime
            Start time
        end_time : datetime
            End time

        Returns
        -------
        dict
            Summary
        """
        duration = end_time - start_time

        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]

        summary = {
            'start_time': str(start_time),
            'end_time': str(end_time),
            'duration_seconds': duration.total_seconds(),
            'duration_formatted': str(duration),
            'n_subjects': len(results),
            'n_successful': len(successful),
            'n_failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'successful_subjects': [r['subject_id'] for r in successful],
            'failed_subjects': [
                {'subject_id': r['subject_id'], 'error': r.get('error', 'Unknown')}
                for r in failed
            ],
            'individual_results': results,
        }

        return summary

    def run(self, data_dir: str, output_dir: str,
            subjects: list = None,
            parallel: bool = True,
            n_jobs: int = None,
            skip_steps: list = None) -> dict:
        """
        Run batch processing.

        Parameters
        ----------
        data_dir : str
            Data directory
        output_dir : str
            Output directory
        subjects : list, optional
            List of subject IDs (if None, process all subjects found in data_dir)
        parallel : bool
            Run in parallel
        n_jobs : int, optional
            Number of parallel jobs
        skip_steps : list, optional
            Steps to skip

        Returns
        -------
        dict
            Batch processing summary
        """
        # Get subject list if not provided
        if subjects is None:
            loader = NiftiDataLoader(data_dir)
            subjects = loader.get_subject_list()

            if not subjects:
                raise ValueError(f"No subjects found in {data_dir}")

        logger.info(f"Found {len(subjects)} subjects to process")

        # Run processing
        if parallel and self.config['parallel'].get('enable', True):
            summary = self.run_parallel(subjects, data_dir, output_dir,
                                       skip_steps, n_jobs)
        else:
            summary = self.run_sequential(subjects, data_dir, output_dir,
                                         skip_steps)

        # Save summary
        output_dir = Path(output_dir)
        summary_file = output_dir / 'batch_processing_summary.json'
        save_metadata(summary, str(summary_file))

        # Print summary
        self._print_summary(summary)

        return summary

    def _print_summary(self, summary: dict):
        """Print batch processing summary."""
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Total subjects:     {summary['n_subjects']}")
        print(f"Successful:         {summary['n_successful']} ({summary['success_rate']:.1f}%)")
        print(f"Failed:             {summary['n_failed']}")
        print(f"Total duration:     {summary['duration_formatted']}")
        print(f"Average per subject: {summary['duration_seconds'] / summary['n_subjects']:.1f} seconds")
        print("="*80)

        if summary['failed_subjects']:
            print("\nFailed subjects:")
            for failed in summary['failed_subjects']:
                print(f"  ✗ {failed['subject_id']}: {failed['error'][:100]}")

        print("\nSuccessful subjects:")
        for subject_id in summary['successful_subjects']:
            print(f"  ✓ {subject_id}")

        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch Processing for Multiple Subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects in parallel
  python batch_processing.py /path/to/data /path/to/output

  # Process specific subjects
  python batch_processing.py /path/to/data /path/to/output --subjects sub-001 sub-002 sub-003

  # Process sequentially
  python batch_processing.py /path/to/data /path/to/output --sequential

  # Limit number of parallel jobs
  python batch_processing.py /path/to/data /path/to/output --n-jobs 4

  # Skip specific pipeline steps
  python batch_processing.py /path/to/data /path/to/output --skip motion_correction
        """
    )

    parser.add_argument('data_dir', help='Data directory containing raw data')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to process')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially (not parallel)')
    parser.add_argument('--n-jobs', type=int, help='Number of parallel jobs')
    parser.add_argument('--skip', nargs='+', help='Steps to skip')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()

    # Run batch processing
    batch = BatchProcessor(args.config)

    try:
        summary = batch.run(
            args.data_dir,
            args.output_dir,
            subjects=args.subjects,
            parallel=not args.sequential,
            n_jobs=args.n_jobs,
            skip_steps=args.skip
        )

        # Exit code based on success
        if summary['n_failed'] == 0:
            print("\n✓ All subjects processed successfully!")
            return 0
        elif summary['n_successful'] > 0:
            print(f"\n⚠ {summary['n_successful']} succeeded, {summary['n_failed']} failed")
            return 1
        else:
            print("\n✗ All subjects failed!")
            return 2

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
