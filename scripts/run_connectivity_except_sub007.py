#!/usr/bin/env python3
"""Run connectivity, ICA, EC, and visualization for all subjects except sub-007."""

import sys
import argparse
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from connectivity.functional_connectivity import FunctionalConnectivity  # noqa: E402
from connectivity.ica_analysis import ICAAnalysis  # noqa: E402
from connectivity.effective_connectivity import EffectiveConnectivity  # noqa: E402
from visualization.effective_connectivity_viz import (  # noqa: E402
    EffectiveConnectivityDirectedViz,
)
from visualization.effective_connectivity_source_sink_viz import (  # noqa: E402
    EffectiveConnectivitySourceSinkViz,
)
from visualization.effective_connectivity_glass_brain_3d import (  # noqa: E402
    EffectiveConnectivityGlassBrain3DViz,
)
from visualization.glass_brain_network import GlassBrainNetworkViz  # noqa: E402
from visualization.activation_patterns import ActivationPatternViz  # noqa: E402
from utils.helpers import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_connectivity_batch")


def discover_subjects(preproc_root: Path) -> list:
    """Return sorted list of subject IDs present in the preprocessed directory."""
    return sorted(
        p.name for p in preproc_root.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def find_preprocessed_func(preproc_root: Path, subject_id: str) -> Path:
    """Find the best available preprocessed functional image for a subject."""
    candidates = [
        preproc_root / subject_id / f"{subject_id}_smoothed.nii.gz",
        preproc_root / subject_id / f"{subject_id}_func_mni.nii.gz",
        preproc_root / subject_id / f"{subject_id}_func_raw.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def process_subject(subject_id: str, func_img: Path, out_root: Path, config: dict) -> dict:
    """Run connectivity->viz steps for a single subject."""
    logger.info(f"\n=== {subject_id}: starting connectivity/ICA/EC ===")

    conn_dir = out_root / "connectivity" / subject_id
    viz_dir = out_root / "visualization" / subject_id
    conn_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fc = FunctionalConnectivity(config)
    fc_results = fc.run(str(func_img), str(conn_dir), subject_id)

    ica = ICAAnalysis(config)
    ica_results = ica.run(str(func_img), str(conn_dir), subject_id)

    ec = EffectiveConnectivity(config)
    ec_results = ec.run(fc_results["time_series_file"], str(conn_dir), subject_id)

    # Directed EC visualization (Granger / TE)
    ec_viz = EffectiveConnectivityDirectedViz(config)
    ec_viz_results = {}
    for method, path in ec_results.get("results_files", {}).items():
        try:
            res = ec_viz.run(path, str(viz_dir), subject_id, method=method)
            ec_viz_results[method] = res
        except Exception as e:
            logger.warning(f"{subject_id}: EC visualization failed for {method}: {e}")

    # EC source/sink visualization (node metrics)
    ss_viz = EffectiveConnectivitySourceSinkViz(config)
    ss_results = {}
    for method, path in ec_results.get("results_files", {}).items():
        try:
            if Path(path).suffix != ".npy":
                continue
            res = ss_viz.run(
                path,
                str(viz_dir),
                subject_id=subject_id,
                method=method,
            )
            ss_results[method] = res
        except Exception as e:
            logger.warning(f"{subject_id}: EC source/sink viz failed for {method}: {e}")

    # EC 3D directed connectome (BrainNet-like)
    ec_3d_viz = EffectiveConnectivityGlassBrain3DViz(config)
    ec_3d_results = {}
    atlas_name = fc_results.get("atlas") or config.get("atlas", {}).get("default", "AAL")
    for method, path in ec_results.get("results_files", {}).items():
        try:
            if Path(path).suffix != ".npy":
                continue
            res = ec_3d_viz.run(
                path,
                str(viz_dir),
                subject_id=subject_id,
                method=method,
                atlas=atlas_name,
            )
            ec_3d_results[method] = res
        except Exception as e:
            logger.warning(f"{subject_id}: EC 3D viz failed for {method}: {e}")

    # Visualization
    corr_path = fc_results["connectivity_matrices"].get("correlation")
    if corr_path:
        viz_net = GlassBrainNetworkViz(config)
        viz_net.run(corr_path, str(viz_dir), subject_id, atlas_name)
    else:
        logger.warning(f"{subject_id}: correlation matrix missing; skipping network viz.")

    viz_act = ActivationPatternViz(config)
    viz_act.run(str(func_img), str(viz_dir), subject_id,
                ica_components=ica_results.get("components_file"))

    logger.info(f"=== {subject_id}: completed ===")

    return {
        "subject_id": subject_id,
        "functional_image": str(func_img),
        "fc": fc_results,
        "ica": ica_results,
        "ec": ec_results,
        "ec_viz": ec_viz_results,
        "ec_source_sink_viz": ss_results,
        "ec_glass_brain_3d_viz": ec_3d_results,
        "status": "completed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run connectivity+ICA+EC+viz for all subjects except sub-007 (preprocessing already done).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preproc-root",
        default="/data/data2/dataset/fpreproc/results/preprocessing",
        help="Directory containing preprocessed outputs (one folder per subject).",
    )
    parser.add_argument(
        "--output-root",
        default="/data/data2/dataset/fpreproc/results",
        help="Root directory for connectivity and visualization outputs.",
    )
    parser.add_argument(
        "--config",
        help="Optional config file path (defaults to config/pipeline_config.yaml).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Explicit subject IDs to process (sub-007 will still be excluded).",
    )
    parser.add_argument(
        "--resume-missing-only",
        action="store_true",
        help="Skip subjects that already have a connectivity folder.",
    )

    args = parser.parse_args()

    preproc_root = Path(args.preproc_root)
    out_root = Path(args.output_root)

    if not preproc_root.exists():
        raise SystemExit(f"Preprocessed root not found: {preproc_root}")

    config = load_config(args.config)

    subjects = args.subjects or discover_subjects(preproc_root)
    subjects = [s for s in subjects if s != "sub-007"]

    if not subjects:
        raise SystemExit("No subjects to process after excluding sub-007.")

    logger.info(f"Subjects to process (excluding sub-007): {subjects}")

    results = []
    for sid in subjects:
        if args.resume_missing_only and (out_root / "connectivity" / sid).exists():
            logger.info(f"{sid}: connectivity output exists, skipping (resume-missing-only).")
            continue

        func_img = find_preprocessed_func(preproc_root, sid)
        if func_img is None:
            logger.warning(f"{sid}: preprocessed functional not found; skipping.")
            continue

        try:
            res = process_subject(sid, func_img, out_root, config)
            results.append(res)
        except Exception as e:
            logger.exception(f"{sid}: failed with error: {e}")
            results.append({"subject_id": sid, "status": "failed", "error": str(e)})

    if any(r.get("status") != "completed" for r in results):
        logger.warning("Some subjects failed. Check logs above.")
        return 1

    logger.info("All subjects processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
