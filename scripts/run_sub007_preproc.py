#!/usr/bin/env python3
"""Run only preprocessing (up to normalization) for sub-007 using /data/data2/dataset/proc data."""

import sys
from pathlib import Path

from pipelines.single_subject import SingleSubjectPipeline


def main():
    """Execute the single-subject run with preprocessing-only steps."""
    data_dir = Path("/data/data2/dataset/proc")
    output_dir = Path("/data/data2/dataset/fpreproc/results")

    pipeline = SingleSubjectPipeline()

    pipeline.run(
        subject_id="sub-007",
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        skip_steps=[
            "functional_connectivity",
            "ica",
            "effective_connectivity",
            "network_viz",
            "activation_viz",
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
