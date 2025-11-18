#!/usr/bin/env python3
"""
Subject-only entry point for the fMRI pipeline.

This wrapper reuses the default SingleSubjectPipeline but automatically
filters out connectivity methods that require multi-subject input (e.g.,
the tangent space embedding). Use this when you want to finish complete
preprocessing and subject-level analyses without triggering the
group-level algorithms.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure sibling imports behave the same way as in single_subject.py
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from single_subject import SingleSubjectPipeline  # noqa: E402


def _filter_group_only_methods(pipeline_config: dict,
                               disallowed: Tuple[str, ...] = ('tangent',)) -> Tuple[List[str], List[str]]:
    """
    Remove group-only connectivity methods from the configuration.

    Parameters
    ----------
    pipeline_config : dict
        Loaded pipeline configuration (modified in place).
    disallowed : tuple of str
        Methods that should never run in the subject-only pipeline.

    Returns
    -------
    tuple(list, list)
        (remaining_methods, removed_methods)
    """
    fc_cfg = pipeline_config.setdefault('connectivity', {}).setdefault('functional', {})
    methods = fc_cfg.get('methods', ['correlation'])
    removed = [m for m in methods if m in disallowed]
    remaining = [m for m in methods if m not in disallowed]

    if not remaining:
        # Ensure we still compute at least correlation matrices
        remaining = ['correlation']

    fc_cfg['methods'] = remaining
    return remaining, removed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the subject-only fMRI pipeline (no group-only connectivity steps)."
    )
    parser.add_argument('subject_id', help="Subject identifier (e.g., sub-001)")
    parser.add_argument('data_dir', help="Directory that contains subject folders")
    parser.add_argument('output_dir', help="Directory where preprocessing results will be written")
    parser.add_argument('--config', help="Optional configuration YAML (defaults to config/pipeline_config.yaml)")
    parser.add_argument('--skip', nargs='+', help="List of pipeline steps to skip (same labels as single_subject.py)")
    parser.add_argument('--start-volume', type=int, metavar='N',
                        help="First fMRI volume to keep (default pulled from the config)")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    skip_steps = args.skip or []

    pipeline = SingleSubjectPipeline(args.config)
    remaining, removed = _filter_group_only_methods(pipeline.config)

    print("=" * 80)
    print("Subject-only pipeline")
    print("=" * 80)
    print(f"Subject: {args.subject_id}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Skip steps: {skip_steps or 'none'}")
    print(f"Connectivity methods: {remaining}")
    if removed:
        print(f"Skipped group-only methods: {removed}")
    print("=" * 80)

    try:
        results = pipeline.run(
            args.subject_id,
            args.data_dir,
            args.output_dir,
            skip_steps=skip_steps,
            start_volume=args.start_volume
        )

        print("\n" + "=" * 80)
        print("✓ SUBJECT PIPELINE COMPLETED")
        print("=" * 80)
        print(f"Subject: {results['subject_id']}")
        print(f"Duration: {results.get('duration_seconds', 0):.1f} seconds")
        print(f"Steps completed: {len(results.get('steps_completed', []))}")
        print("=" * 80)
        return 0

    except Exception as exc:  # pragma: no cover - CLI feedback
        print("\n" + "=" * 80)
        print("✗ SUBJECT PIPELINE FAILED")
        print("=" * 80)
        print(f"Error: {exc}")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
