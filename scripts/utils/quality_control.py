#!/usr/bin/env python3
"""
Quality control utilities for fMRI preprocessing pipeline.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityControl:
    """Quality control metrics and visualizations for fMRI data."""

    def __init__(self, output_dir: str):
        """
        Initialize QC module.

        Parameters
        ----------
        output_dir : str
            Directory for QC outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_motion_metrics(self, motion_params: np.ndarray) -> Dict[str, float]:
        """
        Compute motion quality metrics.

        Parameters
        ----------
        motion_params : numpy.ndarray
            Motion parameters (n_timepoints x 6)

        Returns
        -------
        dict
            Motion metrics
        """
        # Framewise displacement
        motion_mm = motion_params.copy()
        motion_mm[:, 3:] = motion_mm[:, 3:] * 50  # Convert rotations to mm

        fd = np.sum(np.abs(np.diff(motion_mm, axis=0)), axis=1)
        fd = np.concatenate([[0], fd])

        # DVARS (derivative of RMS variance over voxels)
        # This would need the fMRI data - placeholder for now
        metrics = {
            'mean_fd': float(np.mean(fd)),
            'max_fd': float(np.max(fd)),
            'perc_fd_above_0.5': float(np.mean(fd > 0.5) * 100),
            'mean_rotation': float(np.mean(np.abs(motion_params[:, 3:]))),
            'mean_translation': float(np.mean(np.abs(motion_params[:, :3]))),
            'max_rotation': float(np.max(np.abs(motion_params[:, 3:]))),
            'max_translation': float(np.max(np.abs(motion_params[:, :3]))),
        }

        return metrics

    def plot_motion_parameters(self, motion_params: np.ndarray,
                               output_path: str,
                               fd: Optional[np.ndarray] = None) -> None:
        """
        Plot motion parameters.

        Parameters
        ----------
        motion_params : numpy.ndarray
            Motion parameters (n_timepoints x 6)
        output_path : str
            Output file path
        fd : numpy.ndarray, optional
            Framewise displacement
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        timepoints = np.arange(len(motion_params))

        # Translation
        axes[0].plot(timepoints, motion_params[:, 0], label='X')
        axes[0].plot(timepoints, motion_params[:, 1], label='Y')
        axes[0].plot(timepoints, motion_params[:, 2], label='Z')
        axes[0].set_ylabel('Translation (mm)')
        axes[0].set_title('Translation Parameters')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Rotation
        axes[1].plot(timepoints, np.degrees(motion_params[:, 3]), label='Pitch')
        axes[1].plot(timepoints, np.degrees(motion_params[:, 4]), label='Roll')
        axes[1].plot(timepoints, np.degrees(motion_params[:, 5]), label='Yaw')
        axes[1].set_ylabel('Rotation (degrees)')
        axes[1].set_title('Rotation Parameters')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Framewise displacement
        if fd is not None:
            axes[2].plot(timepoints, fd, color='red')
            axes[2].axhline(y=0.5, color='black', linestyle='--', label='0.5mm threshold')
            axes[2].set_ylabel('FD (mm)')
            axes[2].set_title('Framewise Displacement')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        axes[2].set_xlabel('Timepoint')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Motion plot saved: {output_path}")

    def compute_tsnr(self, img: nib.Nifti1Image,
                     mask: Optional[nib.Nifti1Image] = None) -> Tuple[nib.Nifti1Image, float]:
        """
        Compute temporal signal-to-noise ratio (tSNR).

        Parameters
        ----------
        img : nibabel.Nifti1Image
            4D fMRI image
        mask : nibabel.Nifti1Image, optional
            Brain mask

        Returns
        -------
        tsnr_img : nibabel.Nifti1Image
            tSNR map
        mean_tsnr : float
            Mean tSNR within mask
        """
        data = img.get_fdata()

        # Compute tSNR
        mean_signal = np.mean(data, axis=3)
        std_signal = np.std(data, axis=3)

        # Avoid division by zero
        tsnr = np.divide(mean_signal, std_signal,
                        out=np.zeros_like(mean_signal),
                        where=std_signal != 0)

        # Create tSNR image
        tsnr_img = nib.Nifti1Image(tsnr, img.affine)

        # Compute mean tSNR in mask
        if mask is not None:
            mask_data = mask.get_fdata().astype(bool)
            mean_tsnr = float(np.mean(tsnr[mask_data]))
        else:
            mean_tsnr = float(np.mean(tsnr[tsnr > 0]))

        return tsnr_img, mean_tsnr

    def plot_tsnr(self, tsnr_img: nib.Nifti1Image, output_path: str) -> None:
        """
        Plot tSNR map.

        Parameters
        ----------
        tsnr_img : nibabel.Nifti1Image
            tSNR image
        output_path : str
            Output file path
        """
        from nilearn import plotting

        plotting.plot_stat_map(tsnr_img,
                              title='Temporal SNR',
                              display_mode='ortho',
                              cut_coords=None,
                              cmap='hot',
                              output_file=output_path)

        logger.info(f"tSNR plot saved: {output_path}")

    def detect_outliers(self, img: nib.Nifti1Image,
                       mask: nib.Nifti1Image,
                       threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outlier timepoints using DVARS.

        Parameters
        ----------
        img : nibabel.Nifti1Image
            4D fMRI image
        mask : nibabel.Nifti1Image
            Brain mask
        threshold : float
            DVARS threshold (standard deviations)

        Returns
        -------
        dvars : numpy.ndarray
            DVARS values
        outliers : numpy.ndarray
            Boolean array of outlier timepoints
        """
        from nilearn.masking import apply_mask

        # Extract time series
        time_series = apply_mask(img, mask)

        # Compute DVARS
        diff_ts = np.diff(time_series, axis=0)
        dvars = np.sqrt(np.mean(diff_ts ** 2, axis=1))

        # Standardize
        dvars_std = (dvars - np.mean(dvars)) / np.std(dvars)

        # Detect outliers
        outliers = np.abs(dvars_std) > threshold

        # Prepend False for first timepoint
        dvars = np.concatenate([[0], dvars])
        outliers = np.concatenate([[False], outliers])

        return dvars, outliers

    def plot_carpet_plot(self, img: nib.Nifti1Image,
                        mask: nib.Nifti1Image,
                        output_path: str) -> None:
        """
        Create carpet plot (time series heatmap).

        Parameters
        ----------
        img : nibabel.Nifti1Image
            4D fMRI image
        mask : nibabel.Nifti1Image
            Brain mask
        output_path : str
            Output file path
        """
        from nilearn.masking import apply_mask

        # Extract time series
        time_series = apply_mask(img, mask)

        # Subsample voxels for visualization
        n_voxels = min(2000, time_series.shape[1])
        indices = np.random.choice(time_series.shape[1], n_voxels, replace=False)
        time_series_subset = time_series[:, indices]

        # Normalize
        time_series_norm = (time_series_subset - np.mean(time_series_subset, axis=0)) / np.std(time_series_subset, axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(time_series_norm.T, aspect='auto', cmap='gray',
                      interpolation='nearest', vmin=-3, vmax=3)

        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Voxels (sampled)')
        ax.set_title('Carpet Plot')

        plt.colorbar(im, ax=ax, label='Normalized Signal')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Carpet plot saved: {output_path}")

    def create_qc_report(self, subject_id: str,
                        metrics: Dict[str, any],
                        output_path: str) -> None:
        """
        Create HTML QC report.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        metrics : dict
            QC metrics
        output_path : str
            Output HTML file path
        """
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QC Report - {subject_id}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-left: 4px solid #4CAF50;
                    border-radius: 5px;
                    min-width: 200px;
                }}
                .metric-name {{
                    font-weight: bold;
                    color: #666;
                }}
                .metric-value {{
                    font-size: 1.2em;
                    color: #333;
                }}
                .warning {{
                    border-left-color: #ff9800;
                }}
                .error {{
                    border-left-color: #f44336;
                }}
                img {{
                    max-width: 100%;
                    margin: 10px 0;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quality Control Report</h1>
                <p><strong>Subject:</strong> {subject_id}</p>
                <p><strong>Date:</strong> {metrics.get('date', 'N/A')}</p>

                <h2>Motion Metrics</h2>
                <div class="metric">
                    <div class="metric-name">Mean FD</div>
                    <div class="metric-value">{metrics.get('mean_fd', 'N/A'):.3f} mm</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Max FD</div>
                    <div class="metric-value">{metrics.get('max_fd', 'N/A'):.3f} mm</div>
                </div>
                <div class="metric">
                    <div class="metric-name">% FD > 0.5mm</div>
                    <div class="metric-value">{metrics.get('perc_fd_above_0.5', 'N/A'):.1f}%</div>
                </div>

                <h2>Signal Quality</h2>
                <div class="metric">
                    <div class="metric-name">Mean tSNR</div>
                    <div class="metric-value">{metrics.get('mean_tsnr', 'N/A')}</div>
                </div>

                <h2>Figures</h2>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_template)

        logger.info(f"QC report saved: {output_path}")


if __name__ == '__main__':
    print("Quality Control utilities loaded successfully.")
