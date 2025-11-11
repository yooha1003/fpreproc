#!/usr/bin/env python3
"""
3D glass brain network visualization for connectivity matrices.
Optimized for epilepsy research visualization.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlassBrainNetworkViz:
    """3D glass brain network visualization."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize visualization."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.viz_params = config['visualization']['glass_brain']

    def get_roi_coordinates(self, atlas_name: str = 'AAL') -> np.ndarray:
        """
        Get ROI coordinates for atlas.

        Parameters
        ----------
        atlas_name : str
            Atlas name

        Returns
        -------
        numpy.ndarray
            ROI coordinates (n_rois x 3)
        """
        logger.info(f"Getting coordinates for {atlas_name} atlas...")

        from nilearn import datasets
        from nilearn.plotting import find_parcellation_cut_coords

        if atlas_name.upper() == 'AAL':
            atlas = datasets.fetch_atlas_aal()
            atlas_img = atlas['maps']

        elif atlas_name.upper() == 'SCHAEFER':
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
            atlas_img = atlas['maps']

        elif atlas_name.upper() == 'HARVARDOXFORD':
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            atlas_img = atlas['maps']

        else:
            raise ValueError(f"Unknown atlas: {atlas_name}")

        # Get coordinates
        coords = find_parcellation_cut_coords(atlas_img)

        logger.info(f"Found coordinates for {len(coords)} ROIs")

        return coords

    def plot_connectome(self, conn_matrix: np.ndarray, coords: np.ndarray,
                       output_path: str, title: str = 'Connectome',
                       edge_threshold: str = '95%') -> None:
        """
        Plot connectome on glass brain.

        Parameters
        ----------
        conn_matrix : numpy.ndarray
            Connectivity matrix (n_rois x n_rois)
        coords : numpy.ndarray
            ROI coordinates (n_rois x 3)
        output_path : str
            Output file path
        title : str
            Plot title
        edge_threshold : str
            Threshold for edges (e.g., '95%', or float value)
        """
        logger.info(f"Plotting connectome: {title}")

        from nilearn import plotting

        # Create figure
        display_mode = self.viz_params.get('display_mode', 'lyrz')
        colormap = self.viz_params.get('colormap', 'jet')

        # Plot connectome
        fig = plotting.plot_connectome(
            conn_matrix,
            coords,
            edge_threshold=edge_threshold,
            title=title,
            display_mode=display_mode,
            colorbar=True,
            node_size=self.viz_params.get('node_size', 50),
            edge_cmap=colormap,
        )

        # Save
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Connectome saved: {output_path}")

    def plot_connectome_3d(self, conn_matrix: np.ndarray, coords: np.ndarray,
                          output_path: str, title: str = 'Connectome 3D') -> None:
        """
        Plot interactive 3D connectome.

        Parameters
        ----------
        conn_matrix : numpy.ndarray
            Connectivity matrix
        coords : numpy.ndarray
            ROI coordinates
        output_path : str
            Output HTML file path
        title : str
            Plot title
        """
        logger.info("Creating 3D interactive connectome...")

        try:
            import plotly.graph_objects as go

            # Threshold matrix
            threshold = np.percentile(np.abs(conn_matrix), 95)
            conn_thresholded = conn_matrix.copy()
            conn_thresholded[np.abs(conn_thresholded) < threshold] = 0

            # Create edges
            edges_x = []
            edges_y = []
            edges_z = []
            edge_colors = []

            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    if conn_thresholded[i, j] != 0:
                        # Add edge
                        edges_x.extend([coords[i, 0], coords[j, 0], None])
                        edges_y.extend([coords[i, 1], coords[j, 1], None])
                        edges_z.extend([coords[i, 2], coords[j, 2], None])
                        edge_colors.append(conn_thresholded[i, j])

            # Create edge trace
            edge_trace = go.Scatter3d(
                x=edges_x,
                y=edges_y,
                z=edges_z,
                mode='lines',
                line=dict(color='rgba(125,125,125,0.3)', width=1),
                hoverinfo='none',
                name='Connections'
            )

            # Create node trace
            node_trace = go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    line=dict(color='white', width=0.5)
                ),
                text=[f'ROI {i+1}' for i in range(len(coords))],
                hoverinfo='text',
                name='ROIs'
            )

            # Create layout
            layout = go.Layout(
                title=title,
                scene=dict(
                    xaxis=dict(title='X (mm)'),
                    yaxis=dict(title='Y (mm)'),
                    zaxis=dict(title='Z (mm)'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                hovermode='closest'
            )

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

            # Save
            fig.write_html(output_path)

            logger.info(f"3D connectome saved: {output_path}")

        except ImportError:
            logger.warning("Plotly not installed, skipping 3D visualization")

    def plot_circular_connectome(self, conn_matrix: np.ndarray,
                                 output_path: str,
                                 node_labels: Optional[list] = None,
                                 title: str = 'Circular Connectome') -> None:
        """
        Plot circular connectome (chord diagram).

        Parameters
        ----------
        conn_matrix : numpy.ndarray
            Connectivity matrix
        output_path : str
            Output file path
        node_labels : list, optional
            Node labels
        title : str
            Plot title
        """
        logger.info("Creating circular connectome...")

        import matplotlib.pyplot as plt
        from matplotlib.patches import Arc
        import matplotlib.patches as mpatches

        n_nodes = conn_matrix.shape[0]

        if node_labels is None:
            node_labels = [f'{i+1}' for i in range(n_nodes)]

        # Threshold connections
        threshold = np.percentile(np.abs(conn_matrix), 95)
        conn_thresholded = conn_matrix.copy()
        conn_thresholded[np.abs(conn_thresholded) < threshold] = 0

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Node positions on circle
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        # Draw nodes
        ax.scatter(x, y, s=200, c='steelblue', zorder=10)

        # Add labels
        for i, (xi, yi, label) in enumerate(zip(x, y, node_labels)):
            angle_deg = np.degrees(angles[i])
            if angle_deg > 90 and angle_deg < 270:
                ha = 'right'
                angle_deg += 180
            else:
                ha = 'left'

            ax.text(xi * 1.15, yi * 1.15, label, ha=ha,
                   rotation=angle_deg, fontsize=8)

        # Draw connections
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if conn_thresholded[i, j] != 0:
                    # Calculate arc parameters
                    x1, y1 = x[i], y[i]
                    x2, y2 = x[j], y[j]

                    # Draw line (simplified, could be curved)
                    color_intensity = np.abs(conn_thresholded[i, j]) / np.max(np.abs(conn_thresholded))
                    ax.plot([x1, x2], [y1, y2], 'gray', alpha=color_intensity * 0.5,
                           linewidth=1, zorder=1)

        ax.set_title(title, fontsize=16, pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Circular connectome saved: {output_path}")

    def plot_matrix_heatmap(self, conn_matrix: np.ndarray,
                           output_path: str,
                           title: str = 'Connectivity Matrix') -> None:
        """
        Plot connectivity matrix as heatmap.

        Parameters
        ----------
        conn_matrix : numpy.ndarray
            Connectivity matrix
        output_path : str
            Output file path
        title : str
            Plot title
        """
        logger.info("Creating connectivity matrix heatmap...")

        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(conn_matrix, cmap='RdBu_r', center=0,
                   square=True, cbar_kws={'label': 'Connectivity'},
                   ax=ax)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('ROI', fontsize=12)
        ax.set_ylabel('ROI', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Matrix heatmap saved: {output_path}")

    def run(self, conn_matrix_file: str, output_dir: str, subject_id: str,
            atlas_name: str = 'AAL') -> dict:
        """
        Run network visualization pipeline.

        Parameters
        ----------
        conn_matrix_file : str
            Connectivity matrix file (.npy)
        output_dir : str
            Output directory
        subject_id : str
            Subject identifier
        atlas_name : str
            Atlas name

        Returns
        -------
        dict
            Results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load connectivity matrix
        conn_matrix = np.load(conn_matrix_file)
        logger.info(f"Loaded connectivity matrix: {conn_matrix.shape}")

        # Get ROI coordinates
        coords = self.get_roi_coordinates(atlas_name)

        # Adjust if sizes don't match
        if len(coords) != conn_matrix.shape[0]:
            logger.warning(f"Coordinate count ({len(coords)}) doesn't match matrix size ({conn_matrix.shape[0]})")
            # Subsample or pad as needed
            min_size = min(len(coords), conn_matrix.shape[0])
            coords = coords[:min_size]
            conn_matrix = conn_matrix[:min_size, :min_size]

        # Generate visualizations
        results = {}

        # Glass brain connectome
        glass_brain_file = output_dir / f'{subject_id}_connectome_glass_brain.png'
        self.plot_connectome(
            conn_matrix, coords, str(glass_brain_file),
            title=f'Functional Connectome - {subject_id}'
        )
        results['glass_brain'] = str(glass_brain_file)

        # 3D interactive connectome
        connectome_3d_file = output_dir / f'{subject_id}_connectome_3d.html'
        self.plot_connectome_3d(
            conn_matrix, coords, str(connectome_3d_file),
            title=f'3D Connectome - {subject_id}'
        )
        results['connectome_3d'] = str(connectome_3d_file)

        # Circular connectome
        circular_file = output_dir / f'{subject_id}_connectome_circular.png'
        self.plot_circular_connectome(
            conn_matrix, str(circular_file),
            title=f'Circular Connectome - {subject_id}'
        )
        results['circular'] = str(circular_file)

        # Matrix heatmap
        heatmap_file = output_dir / f'{subject_id}_connectivity_matrix.png'
        self.plot_matrix_heatmap(
            conn_matrix, str(heatmap_file),
            title=f'Connectivity Matrix - {subject_id}'
        )
        results['heatmap'] = str(heatmap_file)

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'connectivity_matrix_file': str(conn_matrix_file),
            'atlas': atlas_name,
            'n_rois': conn_matrix.shape[0],
            'visualizations': results,
        }

        metadata_file = output_dir / f'{subject_id}_network_viz_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Network visualization completed for {subject_id}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='3D Glass Brain Network Visualization')
    parser.add_argument('matrix', help='Connectivity matrix file (.npy)')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--atlas', default='AAL', help='Atlas name')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Run visualization
    viz = GlassBrainNetworkViz(config)
    results = viz.run(args.matrix, args.output_dir, args.subject, args.atlas)

    print("\n✓ Network visualization completed!")
    print(f"Glass brain: {results['visualizations']['glass_brain']}")
    print(f"3D connectome: {results['visualizations']['connectome_3d']}")


if __name__ == '__main__':
    main()
