#!/usr/bin/env python3
"""
Directed effective connectivity visualization with arrows.
Generates a circular plot with arrowheads to distinguish FC (undirected)
from EC (directed) results such as Granger causality or transfer entropy.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EffectiveConnectivityDirectedViz:
    """Render directed effective connectivity matrices with arrowheads."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        output_cfg = config.get('output', {})
        self.figure_format = output_cfg.get('figure_format', 'png')
        self.dpi = output_cfg.get('figure_dpi', 300)

    def _select_edges(self, matrix: np.ndarray, top_k: int = 200,
                      min_weight: Optional[float] = None) -> List[Tuple[int, int, float]]:
        """
        Select strongest directed edges from EC matrix.

        Note: matrix[target, source] = influence of source -> target.
        """
        n = matrix.shape[0]
        edges: List[Tuple[int, int, float]] = []

        for tgt in range(n):
            for src in range(n):
                if src == tgt:
                    continue
                weight = float(matrix[tgt, src])

                if np.isnan(weight):
                    continue

                if min_weight is not None and weight < min_weight:
                    continue

                edges.append((src, tgt, weight))

        edges.sort(key=lambda e: abs(e[2]), reverse=True)

        if top_k is not None:
            edges = edges[:top_k]

        return edges

    def plot_directed_circular(self, conn_matrix: np.ndarray, output_path: Path,
                               title: str, top_k: int = 200,
                               min_weight: Optional[float] = None) -> str:
        """Plot directed EC with arrows on a circular layout."""
        n_rois = conn_matrix.shape[0]
        edges = self._select_edges(conn_matrix, top_k=top_k, min_weight=min_weight)

        if not edges:
            logger.warning("No edges passed the selection criteria; nothing to plot.")
            return str(output_path)

        # Circular node coordinates
        angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])

        weights = np.array([abs(e[2]) for e in edges])
        vmax = np.percentile(weights, 95) if weights.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        norm = Normalize(vmin=0, vmax=vmax)
        cmap = plt.get_cmap('plasma')

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw nodes
        ax.scatter(coords[:, 0], coords[:, 1], s=30, color="#222222", zorder=3)
        for idx, (x, y) in enumerate(coords):
            ax.text(x * 1.08, y * 1.08, str(idx + 1),
                    ha='center', va='center', fontsize=6, color="#333333", zorder=4)

        # Draw directed edges with arrowheads
        for src, tgt, weight in edges:
            magnitude = abs(weight)
            color = cmap(norm(magnitude))
            width = 0.5 + 2.5 * norm(magnitude)
            arrow = FancyArrowPatch(
                (coords[src, 0], coords[src, 1]),
                (coords[tgt, 0], coords[tgt, 1]),
                arrowstyle='-|>',
                mutation_scale=8 + 10 * norm(magnitude),
                linewidth=width,
                color=color,
                alpha=0.75,
                shrinkA=6,
                shrinkB=6,
                zorder=2
            )
            ax.add_patch(arrow)

        # Styling
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Influence strength')

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Directed EC visualization saved: {output_path}")
        return str(output_path)

    def run(self, matrix_file: str, output_dir: str, subject_id: str = 'sub-001',
            method: str = 'granger', top_k: int = 200,
            min_weight: Optional[float] = None) -> dict:
        """Generate directed EC visualization."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        conn_matrix = np.load(matrix_file)
        logger.info(f"Loaded EC matrix: {conn_matrix.shape}")

        output_path = output_dir / f"{subject_id}_ec_{method}_directed.{self.figure_format}"
        title = f"Effective Connectivity ({method}) - {subject_id}"

        figure_path = self.plot_directed_circular(
            conn_matrix, output_path, title, top_k=top_k, min_weight=min_weight
        )

        return {
            'subject_id': subject_id,
            'method': method,
            'matrix_file': str(matrix_file),
            'figure': figure_path,
            'top_k_edges': top_k,
            'min_weight': min_weight,
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Directed effective connectivity visualization (arrow plot).'
    )
    parser.add_argument('matrix', help='EC matrix file (.npy)')
    parser.add_argument('output_dir', help='Directory to save the figure')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--method', default='granger',
                        help='Method name to show on the plot (e.g., granger, transfer_entropy)')
    parser.add_argument('--top-k', dest='top_k', type=int, default=200,
                        help='Number of strongest edges to draw (None for all)')
    parser.add_argument('--min-weight', dest='min_weight', type=float, default=None,
                        help='Minimum weight to include an edge')
    parser.add_argument('--config', help='Optional pipeline config file')

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = None

    viz = EffectiveConnectivityDirectedViz(cfg)
    results = viz.run(
        args.matrix,
        args.output_dir,
        subject_id=args.subject,
        method=args.method,
        top_k=args.top_k,
        min_weight=args.min_weight
    )

    print("\n✓ Directed EC visualization created")
    print(f"Figure: {results['figure']}")


if __name__ == '__main__':
    main()
