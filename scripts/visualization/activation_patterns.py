#!/usr/bin/env python3
"""
Brain activation pattern visualization for fMRI data.
Includes tSNR maps, ICA components, and statistical maps.
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


class ActivationPatternViz:
    """Visualization of brain activation patterns."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize visualization."""
        if config is None:
            from utils.helpers import load_config
            config = load_config()

        self.config = config
        self.viz_params = config['visualization']['activation']

    def plot_stat_map(self, img: nib.Nifti1Image, output_path: str,
                     title: str = 'Activation Map',
                     threshold: float = 2.3) -> None:
        """
        Plot statistical map.

        Parameters
        ----------
        img : nibabel.Nifti1Image
            Statistical image
        output_path : str
            Output file path
        title : str
            Plot title
        threshold : float
            Display threshold
        """
        logger.info(f"Plotting statistical map: {title}")

        from nilearn import plotting

        colormap = self.viz_params.get('colormap', 'cold_hot')
        alpha = self.viz_params.get('alpha', 0.7)

        display = plotting.plot_stat_map(
            img,
            title=title,
            threshold=threshold,
            colorbar=True,
            display_mode='ortho',
            cmap=colormap,
            alpha=alpha,
        )

        display.savefig(output_path, dpi=300)
        display.close()

        logger.info(f"Statistical map saved: {output_path}")

    def plot_tsnr_map(self, tsnr_img: nib.Nifti1Image, output_path: str,
                     title: str = 'tSNR Map') -> None:
        """
        Plot temporal SNR map.

        Parameters
        ----------
        tsnr_img : nibabel.Nifti1Image
            tSNR image
        output_path : str
            Output file path
        title : str
            Plot title
        """
        logger.info("Plotting tSNR map...")

        from nilearn import plotting

        display = plotting.plot_stat_map(
            tsnr_img,
            title=title,
            colorbar=True,
            display_mode='ortho',
            cmap='hot',
            vmax=150,
        )

        display.savefig(output_path, dpi=300)
        display.close()

        logger.info(f"tSNR map saved: {output_path}")

    def plot_ica_components(self, components_img: nib.Nifti1Image,
                           output_dir: Path,
                           subject_id: str,
                           n_components: int = None) -> list:
        """
        Plot ICA components.

        Parameters
        ----------
        components_img : nibabel.Nifti1Image
            ICA components image (4D)
        output_dir : Path
            Output directory
        subject_id : str
            Subject identifier
        n_components : int, optional
            Number of components to plot

        Returns
        -------
        list
            List of output files
        """
        logger.info("Plotting ICA components...")

        from nilearn import plotting
        from nilearn.image import index_img

        output_dir.mkdir(parents=True, exist_ok=True)

        if n_components is None:
            n_components = components_img.shape[3]

        output_files = []

        for i in range(n_components):
            component = index_img(components_img, i)

            output_file = output_dir / f'{subject_id}_ica_component_{i:02d}.png'

            display = plotting.plot_stat_map(
                component,
                title=f'ICA Component {i+1}',
                colorbar=True,
                display_mode='z',
                cut_coords=7,
                cmap='cold_hot',
            )

            display.savefig(str(output_file), dpi=150)
            display.close()

            output_files.append(str(output_file))

        logger.info(f"Plotted {n_components} ICA components")

        return output_files

    def plot_mean_functional(self, func_img: nib.Nifti1Image,
                            output_path: str,
                            title: str = 'Mean Functional') -> None:
        """
        Plot mean functional image.

        Parameters
        ----------
        func_img : nibabel.Nifti1Image
            4D functional image
        output_path : str
            Output file path
        title : str
            Plot title
        """
        logger.info("Plotting mean functional...")

        from nilearn import plotting, image

        mean_img = image.mean_img(func_img)

        display = plotting.plot_anat(
            mean_img,
            title=title,
            display_mode='ortho',
            colorbar=True,
        )

        display.savefig(output_path, dpi=300)
        display.close()

        logger.info(f"Mean functional saved: {output_path}")

    def plot_roi_activation(self, stat_img: nib.Nifti1Image,
                           atlas_img: str,
                           output_path: str,
                           title: str = 'ROI Activation') -> None:
        """
        Plot activation with ROI boundaries.

        Parameters
        ----------
        stat_img : nibabel.Nifti1Image
            Statistical image
        atlas_img : str
            Atlas/parcellation image
        output_path : str
            Output file path
        title : str
            Plot title
        """
        logger.info("Plotting ROI activation...")

        from nilearn import plotting

        display = plotting.plot_roi(
            atlas_img,
            bg_img=stat_img,
            title=title,
            display_mode='ortho',
            colorbar=True,
        )

        display.savefig(output_path, dpi=300)
        display.close()

        logger.info(f"ROI activation saved: {output_path}")

    def create_montage(self, img: nib.Nifti1Image, output_path: str,
                      title: str = 'Brain Slices') -> None:
        """
        Create slice montage.

        Parameters
        ----------
        img : nibabel.Nifti1Image
            Brain image
        output_path : str
            Output file path
        title : str
            Plot title
        """
        logger.info("Creating slice montage...")

        from nilearn import plotting

        # Plot all slices
        display = plotting.plot_anat(
            img,
            title=title,
            display_mode='z',
            cut_coords=12,
        )

        display.savefig(output_path, dpi=300)
        display.close()

        logger.info(f"Montage saved: {output_path}")

    def plot_surface_activation(self, stat_img: nib.Nifti1Image,
                                output_path: str,
                                title: str = 'Surface Activation',
                                threshold: float = 2.3) -> None:
        """
        Plot activation on brain surface.

        Parameters
        ----------
        stat_img : nibabel.Nifti1Image
            Statistical image
        output_path : str
            Output file path
        title : str
            Plot title
        threshold : float
            Display threshold
        """
        logger.info("Plotting surface activation...")

        try:
            from nilearn import plotting, datasets

            # Get fsaverage surface
            fsaverage = datasets.fetch_surf_fsaverage()

            # Create figure with multiple views
            fig = plt.figure(figsize=(16, 4))

            views = ['lateral', 'medial']
            hemis = ['left', 'right']

            for i, (hemi, view) in enumerate([(h, v) for h in hemis for v in views]):
                ax = fig.add_subplot(1, 4, i + 1, projection='3d')

                plotting.plot_surf_stat_map(
                    fsaverage[f'pial_{hemi}'],
                    stat_img,
                    hemi=hemi,
                    view=view,
                    threshold=threshold,
                    colorbar=False,
                    axes=ax,
                    title=f'{hemi.capitalize()} - {view.capitalize()}',
                )

            fig.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Surface activation saved: {output_path}")

        except Exception as e:
            logger.warning(f"Surface plotting failed: {e}")

    def run(self, func_img: str, output_dir: str, subject_id: str,
            ica_components: Optional[str] = None) -> dict:
        """
        Run activation pattern visualization pipeline.

        Parameters
        ----------
        func_img : str
            Preprocessed functional image
        output_dir : str
            Output directory
        subject_id : str
            Subject identifier
        ica_components : str, optional
            ICA components image

        Returns
        -------
        dict
            Results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Load functional image
        func = nib.load(func_img)

        # Mean functional
        mean_func_file = output_dir / f'{subject_id}_mean_functional.png'
        self.plot_mean_functional(func, str(mean_func_file),
                                  title=f'Mean Functional - {subject_id}')
        results['mean_functional'] = str(mean_func_file)

        # Compute and plot tSNR
        logger.info("Computing tSNR...")
        from utils.quality_control import QualityControl

        qc = QualityControl(str(output_dir))
        tsnr_img, mean_tsnr = qc.compute_tsnr(func)

        tsnr_file = output_dir / f'{subject_id}_tsnr.nii.gz'
        nib.save(tsnr_img, tsnr_file)

        tsnr_plot_file = output_dir / f'{subject_id}_tsnr_map.png'
        self.plot_tsnr_map(tsnr_img, str(tsnr_plot_file),
                          title=f'tSNR Map - {subject_id} (mean={mean_tsnr:.1f})')
        results['tsnr_map'] = str(tsnr_plot_file)

        # Plot ICA components if provided
        if ica_components and Path(ica_components).exists():
            logger.info(f"Plotting ICA components from {ica_components}")

            ica_img = nib.load(ica_components)
            ica_output_dir = output_dir / 'ica_components'

            ica_files = self.plot_ica_components(
                ica_img, ica_output_dir, subject_id, n_components=10
            )

            results['ica_components'] = ica_files

        # Create montage
        montage_file = output_dir / f'{subject_id}_slice_montage.png'
        self.create_montage(func, str(montage_file),
                           title=f'Brain Slices - {subject_id}')
        results['montage'] = str(montage_file)

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'functional_image': str(func_img),
            'mean_tsnr': float(mean_tsnr),
            'visualizations': results,
        }

        metadata_file = output_dir / f'{subject_id}_activation_viz_metadata.json'
        save_metadata(metadata, str(metadata_file))

        logger.info(f"Activation pattern visualization completed for {subject_id}")

        return metadata


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Activation Pattern Visualization')
    parser.add_argument('input', help='Preprocessed functional image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--subject', default='sub-001', help='Subject ID')
    parser.add_argument('--ica-components', help='ICA components image')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load config
    if args.config:
        from utils.helpers import load_config
        config = load_config(args.config)
    else:
        config = None

    # Run visualization
    viz = ActivationPatternViz(config)
    results = viz.run(args.input, args.output_dir, args.subject,
                     args.ica_components)

    print("\n✓ Activation pattern visualization completed!")
    print(f"Mean tSNR: {results['mean_tsnr']:.1f}")
    print(f"Visualizations: {len(results['visualizations'])}")


if __name__ == '__main__':
    main()
