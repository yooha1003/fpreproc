#!/usr/bin/env python3
"""
Effective Connectivity (EC) analysis for fMRI data.
Implements Granger Causality and Transfer Entropy for epilepsy research.
"""

import sys
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EffectiveConnectivity:
    """Effective connectivity analysis using Granger Causality and Transfer Entropy."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize EC analysis."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.ec_params = config['connectivity']['effective']

    def compute_granger_causality(self, time_series: np.ndarray,
                                  max_lag: int = None) -> np.ndarray:
        """
        Compute pairwise Granger causality.

        Parameters
        ----------
        time_series : numpy.ndarray
            ROI time series (n_timepoints x n_rois)
        max_lag : int, optional
            Maximum lag for GC

        Returns
        -------
        numpy.ndarray
            Granger causality matrix (n_rois x n_rois)
            gc_matrix[i, j] = influence of ROI j on ROI i
        """
        logger.info("Computing Granger causality...")

        if max_lag is None:
            max_lag = self.ec_params.get('max_lag', 5)

        n_rois = time_series.shape[1]
        gc_matrix = np.zeros((n_rois, n_rois))

        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            for i in range(n_rois):
                for j in range(n_rois):
                    if i == j:
                        continue

                    # Test if j Granger-causes i
                    try:
                        data = np.column_stack([time_series[:, i], time_series[:, j]])

                        # Run test
                        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

                        # Get F-statistic for the most significant lag
                        f_stats = [results[lag+1][0]['ssr_ftest'][0] for lag in range(max_lag)]
                        gc_matrix[i, j] = np.max(f_stats)

                    except Exception as e:
                        logger.debug(f"GC test failed for ROI {i}, {j}: {e}")
                        gc_matrix[i, j] = 0

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{n_rois} ROIs")

        except ImportError:
            logger.error("statsmodels not installed, cannot compute Granger causality")
            logger.info("Install with: pip install statsmodels")
            return gc_matrix

        logger.info("Granger causality computation completed")

        return gc_matrix

    def compute_transfer_entropy(self, time_series: np.ndarray,
                                 lag: int = 1) -> np.ndarray:
        """
        Compute transfer entropy between ROIs.

        Parameters
        ----------
        time_series : numpy.ndarray
            ROI time series (n_timepoints x n_rois)
        lag : int
            Time lag

        Returns
        -------
        numpy.ndarray
            Transfer entropy matrix (n_rois x n_rois)
        """
        logger.info("Computing transfer entropy...")

        n_rois = time_series.shape[1]
        te_matrix = np.zeros((n_rois, n_rois))

        # Discretize time series for entropy calculation
        def discretize(x, n_bins=10):
            """Discretize continuous signal."""
            return np.digitize(x, bins=np.linspace(x.min(), x.max(), n_bins))

        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    continue

                try:
                    # Discretize
                    x = discretize(time_series[:, i])
                    y = discretize(time_series[:, j])

                    # Compute TE: I(Y_t ; X_{t-lag} | Y_{t-lag})
                    te = self._compute_te_single(x, y, lag)
                    te_matrix[i, j] = te

                except Exception as e:
                    logger.debug(f"TE computation failed for ROI {i}, {j}: {e}")
                    te_matrix[i, j] = 0

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{n_rois} ROIs")

        logger.info("Transfer entropy computation completed")

        return te_matrix

    def _compute_te_single(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        """
        Compute transfer entropy from x to y.

        Parameters
        ----------
        x : numpy.ndarray
            Source time series (discretized)
        y : numpy.ndarray
            Target time series (discretized)
        lag : int
            Time lag

        Returns
        -------
        float
            Transfer entropy value
        """
        # Prepare lagged series
        n = len(x) - lag

        y_t = y[lag:]
        y_t_lag = y[:n]
        x_t_lag = x[:n]

        # Compute entropies
        h_y_t = self._entropy(y_t)
        h_y_t_y_t_lag = self._conditional_entropy(y_t, y_t_lag)
        h_y_t_y_t_lag_x_t_lag = self._conditional_entropy_3(y_t, y_t_lag, x_t_lag)

        # TE = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
        te = h_y_t_y_t_lag - h_y_t_y_t_lag_x_t_lag

        return te

    def _entropy(self, x: np.ndarray) -> float:
        """Compute Shannon entropy."""
        _, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute conditional entropy H(X|Y)."""
        # H(X|Y) = H(X,Y) - H(Y)
        xy = np.column_stack([x, y])
        xy_tuples = [tuple(row) for row in xy]

        h_xy = self._entropy(np.array(xy_tuples, dtype=object))
        h_y = self._entropy(y)

        return h_xy - h_y

    def _conditional_entropy_3(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Compute conditional entropy H(X|Y,Z)."""
        # H(X|Y,Z) = H(X,Y,Z) - H(Y,Z)
        xyz = np.column_stack([x, y, z])
        yz = np.column_stack([y, z])

        xyz_tuples = [tuple(row) for row in xyz]
        yz_tuples = [tuple(row) for row in yz]

        h_xyz = self._entropy(np.array(xyz_tuples, dtype=object))
        h_yz = self._entropy(np.array(yz_tuples, dtype=object))

        return h_xyz - h_yz

    def compute_spectral_granger(self, time_series: np.ndarray,
                                fs: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Compute spectral Granger causality (frequency-resolved).

        Parameters
        ----------
        time_series : numpy.ndarray
            ROI time series
        fs : float
            Sampling frequency (Hz)

        Returns
        -------
        dict
            Spectral GC results
        """
        logger.info("Computing spectral Granger causality...")

        try:
            from mne_connectivity import spectral_connectivity_epochs

            # Reshape for MNE: (n_epochs, n_signals, n_times)
            # For simplicity, treat as single epoch
            data = time_series.T[np.newaxis, :, :]

            # Compute spectral connectivity
            con = spectral_connectivity_epochs(
                data,
                method='gc',
                mode='multitaper',
                sfreq=fs,
                fmin=0.01,
                fmax=0.1,
                faverage=False,
                verbose=False
            )

            results = {
                'connectivity': con.get_data(),
                'freqs': con.freqs,
            }

            logger.info("Spectral Granger causality completed")

            return results

        except ImportError:
            logger.warning("mne-connectivity not installed, skipping spectral GC")
            return {}

        except Exception as e:
            logger.warning(f"Spectral GC failed: {e}")
            return {}

    def run(self, time_series_file: str, output_dir: str, subject_id: str) -> dict:
        """
        Run effective connectivity analysis.

        Parameters
        ----------
        time_series_file : str
            ROI time series file (.npy)
        output_dir : str
            Output directory
        subject_id : str
            Subject identifier

        Returns
        -------
        dict
            Results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load time series
        time_series = np.load(time_series_file)
        logger.info(f"Loaded time series: {time_series.shape}")

        methods = self.ec_params.get('methods', ['granger'])

        results = {}

        # Granger Causality
        if 'granger' in methods:
            gc_matrix = self.compute_granger_causality(time_series)
            gc_file = output_dir / f'{subject_id}_ec_granger.npy'
            np.save(gc_file, gc_matrix)
            results['granger'] = str(gc_file)

        # Transfer Entropy
        if 'transfer_entropy' in methods:
            te_matrix = self.compute_transfer_entropy(time_series)
            te_file = output_dir / f'{subject_id}_ec_transfer_entropy.npy'
            np.save(te_file, te_matrix)
            results['transfer_entropy'] = str(te_file)

        # Spectral Granger (if MNE available)
        spectral_results = self.compute_spectral_granger(time_series)
        if spectral_results:
            spec_file = output_dir / f'{subject_id}_ec_spectral_granger.npz'
            np.savez(spec_file, **spectral_results)
            results['spectral_granger'] = str(spec_file)

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'time_series_file': str(time_series_file),
            'n_rois': time_series.shape[1],
            'n_timepoints': time_series.shape[0],
            'methods': methods,
            'results_files': results,
        }

        metadata_file = output_dir / f'{subject_id}_ec_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Effective connectivity analysis completed for {subject_id}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Effective Connectivity Analysis')
    parser.add_argument('timeseries', help='ROI time series file (.npy)')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Run EC analysis
    ec = EffectiveConnectivity(config)
    results = ec.run(args.timeseries, args.output_dir, args.subject)

    print("\n✓ Effective connectivity analysis completed!")
    print(f"Methods: {', '.join(results['methods'])}")
    print(f"Number of ROIs: {results['n_rois']}")


if __name__ == '__main__':
    main()
