#!/usr/bin/env python3
"""
Effective Connectivity (EC) source/sink visualization.

Computes node-level directed metrics from an EC matrix:
  - in-strength:  sum of incoming influence to ROI k
  - out-strength: sum of outgoing influence from ROI k
  - net-flow:     out - in (source-like if positive, sink-like if negative)

Assumes EC matrix convention used in this repo:
  matrix[target, source] = influence of source -> target
"""

import sys
import csv
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EffectiveConnectivitySourceSinkViz:
    """Compute and visualize EC source/sink node metrics."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        output_cfg = config.get("output", {})
        self.figure_format = output_cfg.get("figure_format", "png")
        self.dpi = output_cfg.get("figure_dpi", 300)

    def _load_roi_labels(self, labels_file: Optional[str], n_rois: int) -> List[str]:
        if not labels_file:
            return [f"ROI {i + 1:03d}" for i in range(n_rois)]

        path = Path(labels_file)
        if not path.exists():
            raise FileNotFoundError(f"ROI labels file not found: {path}")

        labels: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                label = line.strip()
                if label:
                    labels.append(label)

        if not labels:
            logger.warning("ROI labels file is empty; falling back to numeric labels.")
            return [f"ROI {i + 1:03d}" for i in range(n_rois)]

        if len(labels) != n_rois:
            logger.warning(
                f"ROI label count ({len(labels)}) != n_rois ({n_rois}); "
                "trimming/padding to match matrix size."
            )

        if len(labels) > n_rois:
            labels = labels[:n_rois]
        else:
            labels = labels + [f"ROI {i + 1:03d}" for i in range(len(labels), n_rois)]

        return labels

    def _prepare_matrix(
        self,
        matrix: np.ndarray,
        *,
        top_k: Optional[int],
        min_weight: Optional[float],
        use_abs: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"EC matrix must be square 2D array, got shape={matrix.shape}")

        prepared = np.array(matrix, dtype=float, copy=True)
        prepared[~np.isfinite(prepared)] = 0.0
        np.fill_diagonal(prepared, 0.0)

        n = prepared.shape[0]
        mask = ~np.eye(n, dtype=bool)

        magnitude = np.abs(prepared)

        if min_weight is not None:
            mask &= magnitude >= float(min_weight)

        if top_k is not None:
            top_k = int(top_k)
            if top_k <= 0:
                raise ValueError("--top-k must be a positive integer or omitted")

            eligible = magnitude[mask]
            if eligible.size > top_k:
                kth = np.partition(eligible, -top_k)[-top_k]
                mask &= magnitude >= kth

        prepared = np.where(mask, prepared, 0.0)

        if use_abs:
            prepared = np.abs(prepared)

        stats = {
            "n_rois": n,
            "n_nonzero_edges": int(np.count_nonzero(prepared)),
            "top_k": top_k,
            "min_weight": min_weight,
            "use_abs": use_abs,
        }
        return prepared, stats

    def compute_node_metrics(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute node metrics under convention matrix[target, source] = source -> target.
        """
        out_strength = matrix.sum(axis=0)
        in_strength = matrix.sum(axis=1)
        net_flow = out_strength - in_strength

        return {
            "in_strength": in_strength,
            "out_strength": out_strength,
            "net_flow": net_flow,
        }

    def _select_top_sources_sinks(
        self,
        net_flow: np.ndarray,
        top_n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        top_n = int(top_n)
        if top_n <= 0:
            raise ValueError("--top-n must be a positive integer")

        sources = np.where(net_flow > 0)[0]
        sinks = np.where(net_flow < 0)[0]

        if sources.size:
            sources = sources[np.argsort(net_flow[sources])[::-1]]
        else:
            sources = np.argsort(net_flow)[::-1]

        if sinks.size:
            sinks = sinks[np.argsort(net_flow[sinks])]
        else:
            sinks = np.argsort(net_flow)

        return sources[:top_n], sinks[:top_n]

    def plot(
        self,
        matrix: np.ndarray,
        metrics: Dict[str, np.ndarray],
        labels: List[str],
        output_path: Path,
        *,
        title: str,
        top_n: int,
    ) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        in_strength = metrics["in_strength"]
        out_strength = metrics["out_strength"]
        net_flow = metrics["net_flow"]

        source_idx, sink_idx = self._select_top_sources_sinks(net_flow, top_n=top_n)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        ax_src, ax_sink = axes[0, 0], axes[0, 1]
        ax_scatter, ax_mat = axes[1, 0], axes[1, 1]

        # Top sources (net-flow > 0)
        src_vals = net_flow[source_idx]
        src_labels = [labels[i] for i in source_idx]
        ax_src.barh(src_labels[::-1], src_vals[::-1], color="#d62728", alpha=0.85)
        ax_src.set_title("Top sources (net-flow)")
        ax_src.set_xlabel("out - in")
        ax_src.grid(True, axis="x", linestyle="--", alpha=0.3)

        # Top sinks (net-flow < 0)
        sink_vals = net_flow[sink_idx]
        sink_labels = [labels[i] for i in sink_idx]
        ax_sink.barh(sink_labels[::-1], sink_vals[::-1], color="#1f77b4", alpha=0.85)
        ax_sink.set_title("Top sinks (net-flow)")
        ax_sink.set_xlabel("out - in")
        ax_sink.grid(True, axis="x", linestyle="--", alpha=0.3)

        # Scatter: out vs in colored by net-flow
        vmax = np.percentile(np.abs(net_flow), 95) if net_flow.size else 1.0
        vmax = float(vmax) if vmax and vmax > 0 else 1.0
        sc = ax_scatter.scatter(
            in_strength,
            out_strength,
            c=net_flow,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            s=22,
            alpha=0.85,
            edgecolors="none",
        )
        max_axis = float(max(np.max(in_strength), np.max(out_strength), 1.0))
        ax_scatter.plot([0, max_axis], [0, max_axis], color="#444444", lw=1, alpha=0.6)
        ax_scatter.set_xlim(0, max_axis * 1.02)
        ax_scatter.set_ylim(0, max_axis * 1.02)
        ax_scatter.set_xlabel("In-strength")
        ax_scatter.set_ylabel("Out-strength")
        ax_scatter.set_title("Node strengths (color = net-flow)")
        cbar = fig.colorbar(sc, ax=ax_scatter, fraction=0.046, pad=0.04)
        cbar.set_label("Net-flow (out - in)")

        # Annotate a few extreme nodes
        annotate_n = min(3, source_idx.size)
        for idx in source_idx[:annotate_n]:
            ax_scatter.annotate(
                labels[idx],
                (in_strength[idx], out_strength[idx]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#d62728",
            )

        annotate_n = min(3, sink_idx.size)
        for idx in sink_idx[:annotate_n]:
            ax_scatter.annotate(
                labels[idx],
                (in_strength[idx], out_strength[idx]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#1f77b4",
            )

        # EC matrix heatmap (sorted by net-flow)
        order = np.argsort(net_flow)[::-1]
        matrix_sorted = matrix[order][:, order]

        nonzero = matrix_sorted[np.nonzero(matrix_sorted)]
        if nonzero.size:
            if np.min(nonzero) < 0 and np.max(nonzero) > 0:
                vmax_mat = np.percentile(np.abs(nonzero), 99)
                vmax_mat = float(vmax_mat) if vmax_mat and vmax_mat > 0 else 1.0
                vmin_mat = -vmax_mat
                cmap = "coolwarm"
            else:
                vmax_mat = np.percentile(nonzero, 99)
                vmax_mat = float(vmax_mat) if vmax_mat and vmax_mat > 0 else 1.0
                vmin_mat = 0.0
                cmap = "plasma"
        else:
            vmin_mat, vmax_mat, cmap = 0.0, 1.0, "plasma"

        im = ax_mat.imshow(matrix_sorted, interpolation="nearest", cmap=cmap, vmin=vmin_mat, vmax=vmax_mat)
        ax_mat.set_title("EC matrix (sorted by net-flow)")
        ax_mat.set_xlabel("Source (sorted)")
        ax_mat.set_ylabel("Target (sorted)")
        ax_mat.set_xticks([])
        ax_mat.set_yticks([])
        cbar2 = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
        cbar2.set_label("Influence")

        fig.suptitle(title, fontsize=14)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"EC source/sink visualization saved: {output_path}")
        return str(output_path)

    def _write_metrics_csv(self, metrics: Dict[str, np.ndarray], labels: List[str], csv_path: Path) -> str:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        in_strength = metrics["in_strength"]
        out_strength = metrics["out_strength"]
        net_flow = metrics["net_flow"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["roi_index", "label", "in_strength", "out_strength", "net_flow"])
            for i, label in enumerate(labels):
                writer.writerow([i + 1, label, float(in_strength[i]), float(out_strength[i]), float(net_flow[i])])

        return str(csv_path)

    def run(
        self,
        matrix_file: str,
        output_dir: str,
        *,
        subject_id: str = "sub-001",
        method: str = "granger",
        top_n: int = 10,
        top_k: Optional[int] = None,
        min_weight: Optional[float] = None,
        use_abs: bool = False,
        roi_labels: Optional[str] = None,
    ) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        matrix_file_path = Path(matrix_file)
        if not matrix_file_path.exists():
            raise FileNotFoundError(f"EC matrix file not found: {matrix_file_path}")

        raw_matrix = np.load(matrix_file_path)
        logger.info(f"Loaded EC matrix: {raw_matrix.shape}")

        matrix, prep_stats = self._prepare_matrix(
            raw_matrix, top_k=top_k, min_weight=min_weight, use_abs=use_abs
        )
        labels_list = self._load_roi_labels(roi_labels, matrix.shape[0])
        metrics = self.compute_node_metrics(matrix)

        figure_path = output_dir / f"{subject_id}_ec_{method}_source_sink.{self.figure_format}"
        title = f"EC Source/Sink Metrics ({method}) - {subject_id}"
        fig_out = self.plot(matrix, metrics, labels_list, figure_path, title=title, top_n=top_n)

        csv_path = output_dir / f"{subject_id}_ec_{method}_node_metrics.csv"
        csv_out = self._write_metrics_csv(metrics, labels_list, csv_path)

        metadata = {
            "subject_id": subject_id,
            "method": method,
            "matrix_file": str(matrix_file),
            "figure": fig_out,
            "node_metrics_csv": csv_out,
            "top_n": int(top_n),
            **prep_stats,
        }

        metadata_file = output_dir / f"{subject_id}_ec_{method}_source_sink_metadata.json"
        save_metadata(metadata, str(metadata_file))
        metadata["metadata_file"] = str(metadata_file)

        return metadata


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Effective connectivity source/sink visualization (node metrics)."
    )
    parser.add_argument("matrix", help="EC matrix file (.npy)")
    parser.add_argument("output_dir", help="Directory to save outputs")
    parser.add_argument("--subject", default="sub-001", help="Subject ID")
    parser.add_argument(
        "--method",
        default="granger",
        help="Method name to show on the plot (e.g., granger, transfer_entropy)",
    )
    parser.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=10,
        help="Number of top sources and sinks to display",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Keep only the top-k strongest edges (by |weight|) before computing metrics",
    )
    parser.add_argument(
        "--min-weight",
        dest="min_weight",
        type=float,
        default=None,
        help="Minimum |weight| threshold to keep an edge before computing metrics",
    )
    parser.add_argument(
        "--use-abs",
        action="store_true",
        help="Use absolute weights for metrics (ignores direction sign)",
    )
    parser.add_argument(
        "--roi-labels",
        help="Optional text file with one ROI label per line (must match matrix size)",
    )
    parser.add_argument("--config", help="Optional pipeline config file (for figure format/dpi)")

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = None

    viz = EffectiveConnectivitySourceSinkViz(cfg)
    results = viz.run(
        args.matrix,
        args.output_dir,
        subject_id=args.subject,
        method=args.method,
        top_n=args.top_n,
        top_k=args.top_k,
        min_weight=args.min_weight,
        use_abs=args.use_abs,
        roi_labels=args.roi_labels,
    )

    print("\n✓ EC source/sink visualization created")
    print(f"Figure: {results['figure']}")
    print(f"Node metrics CSV: {results['node_metrics_csv']}")


if __name__ == "__main__":
    main()
