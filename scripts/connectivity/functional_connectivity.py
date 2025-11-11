#!/usr/bin/env python3
"""
Functional Connectivity (FC) analysis for fMRI data.
Optimized for epilepsy research.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FunctionalConnectivity:
    """Functional connectivity analysis."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize FC analysis."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.fc_params = config['connectivity']['functional']
        self.atlas_params = config['atlas']

    def extract_roi_time_series(self, func_img: str, atlas_img: str) -> Tuple[np.ndarray, list]:
        """
        Extract ROI time series from functional data using atlas.

        Parameters
        ----------
        func_img : str
            Preprocessed functional image (4D)
        atlas_img : str
            Atlas/parcellation image

        Returns
        -------
        time_series : numpy.ndarray
            ROI time series (n_timepoints x n_rois)
        roi_labels : list
            ROI labels/names
        """
        logger.info("Extracting ROI time series...")

        from nilearn.maskers import NiftiLabelsMasker

        # Create masker
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize=True,
            detrend=True,
            verbose=0
        )

        # Extract time series
        time_series = masker.fit_transform(func_img)

        logger.info(f"Extracted time series: {time_series.shape}")

        # Get ROI labels
        n_rois = time_series.shape[1]
        roi_labels = [f'ROI_{i+1}' for i in range(n_rois)]

        return time_series, roi_labels

    def compute_correlation_matrix(self, time_series: np.ndarray,
                                   method: str = 'correlation') -> np.ndarray:
        """
        Compute functional connectivity matrix.

        Parameters
        ----------
        time_series : numpy.ndarray
            ROI time series (n_timepoints x n_rois)
        method : str
            Connectivity method ('correlation', 'partial_correlation', 'tangent')

        Returns
        -------
        numpy.ndarray
            Connectivity matrix (n_rois x n_rois)
        """
        logger.info(f"Computing connectivity matrix ({method})...")

        from nilearn.connectome import ConnectivityMeasure

        conn_measure = ConnectivityMeasure(kind=method)
        conn_matrix = conn_measure.fit_transform([time_series])[0]

        logger.info(f"Connectivity matrix shape: {conn_matrix.shape}")

        return conn_matrix

    def threshold_matrix(self, matrix: np.ndarray, threshold: float) -> np.ndarray:
        """
        Threshold connectivity matrix.

        Parameters
        ----------
        matrix : numpy.ndarray
            Connectivity matrix
        threshold : float
            Threshold value

        Returns
        -------
        numpy.ndarray
            Thresholded matrix
        """
        thresholded = matrix.copy()
        thresholded[np.abs(thresholded) < threshold] = 0

        return thresholded

    def compute_graph_metrics(self, conn_matrix: np.ndarray) -> Dict[str, any]:
        """
        Compute graph theory metrics.

        Parameters
        ----------
        conn_matrix : numpy.ndarray
            Connectivity matrix

        Returns
        -------
        dict
            Graph metrics
        """
        logger.info("Computing graph theory metrics...")

        import networkx as nx
        from bct import clustering_coef_bu, modularity_und, betweenness_bin

        # Create binary graph
        threshold = self.fc_params.get('threshold', 0.3)
        binary_matrix = (np.abs(conn_matrix) > threshold).astype(int)

        # Remove self-connections
        np.fill_diagonal(binary_matrix, 0)

        # Create NetworkX graph
        G = nx.from_numpy_array(binary_matrix)

        # Compute metrics
        metrics = {}

        # Basic metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)

        # Node-level metrics
        try:
            metrics['degree'] = dict(G.degree())
            metrics['clustering'] = nx.clustering(G)
            metrics['betweenness'] = nx.betweenness_centrality(G)
        except:
            logger.warning("Some node metrics could not be computed")

        # Global metrics
        try:
            if nx.is_connected(G):
                metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
                metrics['global_efficiency'] = nx.global_efficiency(G)
            else:
                logger.warning("Graph is not connected, some metrics unavailable")
                # Compute for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                G_cc = G.subgraph(largest_cc)
                metrics['avg_shortest_path'] = nx.average_shortest_path_length(G_cc)
                metrics['global_efficiency'] = nx.global_efficiency(G_cc)

            metrics['avg_clustering'] = nx.average_clustering(G)
            metrics['transitivity'] = nx.transitivity(G)

        except Exception as e:
            logger.warning(f"Could not compute global metrics: {e}")

        # Brain Connectivity Toolbox metrics
        try:
            # Clustering coefficient
            clustering_bct = clustering_coef_bu(binary_matrix)
            metrics['clustering_bct'] = clustering_bct.tolist()

            # Modularity
            from bct import community_louvain
            ci, q = community_louvain(binary_matrix)
            metrics['modularity'] = float(q)
            metrics['community_structure'] = ci.tolist()

        except Exception as e:
            logger.warning(f"BCT metrics failed: {e}")

        logger.info("Graph metrics computed successfully")

        return metrics

    def get_atlas(self, atlas_name: str = None) -> str:
        """
        Get atlas file path.

        Parameters
        ----------
        atlas_name : str, optional
            Atlas name

        Returns
        -------
        str
            Atlas file path
        """
        from nilearn import datasets

        if atlas_name is None:
            atlas_name = self.atlas_params.get('default', 'AAL')

        logger.info(f"Loading atlas: {atlas_name}")

        if atlas_name.upper() == 'AAL':
            atlas = datasets.fetch_atlas_aal()
            return atlas['maps']

        elif atlas_name.upper() == 'SCHAEFER':
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
            return atlas['maps']

        elif atlas_name.upper() == 'HARVARDOXFORD':
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            return atlas['maps']

        else:
            raise ValueError(f"Unknown atlas: {atlas_name}")

    def run(self, func_img: str, output_dir: str, subject_id: str,
            atlas_name: str = None) -> dict:
        """
        Run functional connectivity analysis.

        Parameters
        ----------
        func_img : str
            Preprocessed functional image
        output_dir : str
            Output directory
        subject_id : str
            Subject identifier
        atlas_name : str, optional
            Atlas name

        Returns
        -------
        dict
            Results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get atlas
        atlas_img = self.get_atlas(atlas_name)

        # Extract ROI time series
        time_series, roi_labels = self.extract_roi_time_series(func_img, atlas_img)

        # Save time series
        ts_file = output_dir / f'{subject_id}_roi_timeseries.npy'
        np.save(ts_file, time_series)

        # Compute connectivity matrices for different methods
        methods = self.fc_params.get('methods', ['correlation'])

        conn_matrices = {}
        for method in methods:
            logger.info(f"Computing {method} connectivity...")
            conn_matrix = self.compute_correlation_matrix(time_series, method)
            conn_matrices[method] = conn_matrix

            # Save matrix
            matrix_file = output_dir / f'{subject_id}_fc_{method}.npy'
            np.save(matrix_file, conn_matrix)

        # Compute graph metrics (using correlation)
        primary_matrix = conn_matrices.get('correlation', list(conn_matrices.values())[0])
        graph_metrics = self.compute_graph_metrics(primary_matrix)

        # Save graph metrics
        import json

        metrics_file = output_dir / f'{subject_id}_graph_metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types
            metrics_serializable = {}
            for k, v in graph_metrics.items():
                if isinstance(v, np.ndarray):
                    metrics_serializable[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    metrics_serializable[k] = float(v)
                elif isinstance(v, dict):
                    metrics_serializable[k] = {str(k2): float(v2) for k2, v2 in v.items()}
                else:
                    metrics_serializable[k] = v

            json.dump(metrics_serializable, f, indent=2)

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'functional_image': str(func_img),
            'atlas': atlas_name or self.atlas_params.get('default', 'AAL'),
            'n_rois': len(roi_labels),
            'n_timepoints': time_series.shape[0],
            'time_series_file': str(ts_file),
            'connectivity_matrices': {m: str(output_dir / f'{subject_id}_fc_{m}.npy')
                                     for m in methods},
            'graph_metrics_file': str(metrics_file),
            'graph_metrics_summary': {
                'n_nodes': graph_metrics.get('n_nodes'),
                'n_edges': graph_metrics.get('n_edges'),
                'density': graph_metrics.get('density'),
                'modularity': graph_metrics.get('modularity'),
            }
        }

        metadata_file = output_dir / f'{subject_id}_fc_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Functional connectivity analysis completed for {subject_id}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Functional Connectivity Analysis')
    parser.add_argument('input', help='Preprocessed functional image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--atlas', help='Atlas name (AAL, Schaefer, HarvardOxford)')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Run FC analysis
    fc = FunctionalConnectivity(config)
    results = fc.run(args.input, args.output_dir, args.subject, args.atlas)

    print("\n✓ Functional connectivity analysis completed!")
    print(f"Number of ROIs: {results['n_rois']}")
    print(f"Graph density: {results['graph_metrics_summary']['density']:.3f}")
    if results['graph_metrics_summary'].get('modularity'):
        print(f"Modularity: {results['graph_metrics_summary']['modularity']:.3f}")


if __name__ == '__main__':
    main()
