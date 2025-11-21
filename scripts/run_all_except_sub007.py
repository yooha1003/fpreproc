#!/usr/bin/env python3
"""Batch run for all subjects except sub-007."""

import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.batch_processing import BatchProcessor  # noqa: E402
from scripts.utils.data_loader import NiftiDataLoader  # noqa: E402


def main() -> int:
    """Run batch processing for every subject except sub-007."""
    parser = argparse.ArgumentParser(
        description="Process all subjects except sub-007",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="/data/data2/dataset/proc",
        help="Input data root containing subject folders",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/data2/dataset/fpreproc/results",
        help="Output directory for results",
    )
    parser.add_argument("--config", help="Optional config file path")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run subjects one at a time instead of in parallel",
    )
    parser.add_argument("--n-jobs", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--skip",
        nargs="+",
        help="Pipeline steps to skip (e.g., functional_connectivity ica)",
    )
    parser.add_argument(
        "--start-volume",
        type=int,
        metavar="N",
        help="First volume to keep (1-indexed). Default uses config.",
    )
    args = parser.parse_args()

    loader = NiftiDataLoader(args.data_dir)
    subjects = [s for s in loader.get_subject_list() if s != "sub-007"]
    if not subjects:
        raise SystemExit("No subjects found after excluding sub-007.")

    batch = BatchProcessor(args.config)
    summary = batch.run(
        args.data_dir,
        args.output_dir,
        subjects=subjects,
        parallel=not args.sequential,
        n_jobs=args.n_jobs,
        skip_steps=args.skip,
        start_volume=args.start_volume,
    )

    return 0 if summary.get("n_failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
