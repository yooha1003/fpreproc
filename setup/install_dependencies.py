#!/usr/bin/env python3
"""
Automated dependency installation script for fMRI preprocessing pipeline.
This script checks for required software and installs missing Python packages.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class DependencyInstaller:
    """Manages installation and verification of pipeline dependencies."""

    def __init__(self):
        self.missing_software = []
        self.missing_packages = []

    def check_command(self, command):
        """Check if a command is available in PATH."""
        try:
            subprocess.run([command, '--version'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         check=False)
            return True
        except FileNotFoundError:
            return False

    def check_software_dependencies(self):
        """Check for required neuroimaging software."""
        print("\n" + "="*70)
        print("Checking Neuroimaging Software Dependencies")
        print("="*70 + "\n")

        software = {
            'FSL': ['fsl', 'flirt', 'bet'],
            'ANTs': ['antsRegistration', 'antsApplyTransforms'],
            'AFNI': ['3dvolreg', '3dTshift'],
            'FreeSurfer': ['mri_convert', 'recon-all'],
        }

        for package, commands in software.items():
            found = False
            for cmd in commands:
                if self.check_command(cmd):
                    print(f"✓ {package:15s} - Found ({cmd})")
                    found = True
                    break

            if not found:
                print(f"✗ {package:15s} - NOT FOUND (Optional but recommended)")
                self.missing_software.append(package)

        # Check for fMRIPrep (Docker/Singularity based)
        if self.check_command('fmriprep'):
            print(f"✓ {'fMRIPrep':15s} - Found")
        elif self.check_command('docker'):
            print(f"✓ {'fMRIPrep':15s} - Docker available (can run fMRIPrep)")
        else:
            print(f"✗ {'fMRIPrep':15s} - NOT FOUND (Optional)")
            self.missing_software.append('fMRIPrep')

    def install_python_packages(self):
        """Install required Python packages from requirements.txt."""
        print("\n" + "="*70)
        print("Installing Python Dependencies")
        print("="*70 + "\n")

        requirements_file = Path(__file__).parent.parent / 'requirements.txt'

        if not requirements_file.exists():
            print(f"✗ Requirements file not found: {requirements_file}")
            return False

        try:
            print(f"Installing packages from {requirements_file}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])

            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ])

            print("\n✓ Python packages installed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error installing Python packages: {e}")
            return False

    def verify_python_imports(self):
        """Verify that key Python packages can be imported."""
        print("\n" + "="*70)
        print("Verifying Python Package Imports")
        print("="*70 + "\n")

        packages = [
            'nibabel',
            'nilearn',
            'nipype',
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'seaborn',
            'networkx',
            'sklearn',
        ]

        failed = []
        for package in packages:
            try:
                __import__(package)
                print(f"✓ {package:20s} - OK")
            except ImportError:
                print(f"✗ {package:20s} - FAILED")
                failed.append(package)

        if failed:
            print(f"\n✗ Failed to import: {', '.join(failed)}")
            return False
        else:
            print("\n✓ All key packages verified!")
            return True

    def download_atlases(self):
        """Download common brain atlases if not present."""
        print("\n" + "="*70)
        print("Setting up Brain Atlases")
        print("="*70 + "\n")

        atlas_dir = Path(__file__).parent.parent / 'config' / 'atlas'
        atlas_dir.mkdir(parents=True, exist_ok=True)

        print(f"Atlas directory: {atlas_dir}")
        print("\nAtlases can be downloaded from:")
        print("  - AAL: https://www.gin.cnrs.fr/en/tools/aal/")
        print("  - Schaefer: https://github.com/ThomasYeoLab/CBIG")
        print("  - Power: https://www.jonathanpower.net/2011-neuron-bigbrain.html")
        print("  - Harvard-Oxford: Included with FSL")

        print("\n✓ Using nilearn's built-in atlas fetching capabilities")

        try:
            from nilearn import datasets

            # Download common atlases
            print("\nDownloading atlases via nilearn...")

            # AAL
            print("  - Fetching AAL atlas...")
            aal = datasets.fetch_atlas_aal()
            print(f"    ✓ AAL atlas: {aal['maps']}")

            # Harvard-Oxford
            print("  - Fetching Harvard-Oxford atlas...")
            harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            print(f"    ✓ Harvard-Oxford atlas: {harvard_oxford['maps']}")

            # Schaefer
            print("  - Fetching Schaefer atlas...")
            schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
            print(f"    ✓ Schaefer atlas: {schaefer['maps']}")

            print("\n✓ Atlases downloaded successfully!")
            return True

        except Exception as e:
            print(f"\n✗ Error downloading atlases: {e}")
            print("  You can manually download atlases later if needed.")
            return False

    def create_directory_structure(self):
        """Create necessary directories for the pipeline."""
        print("\n" + "="*70)
        print("Creating Directory Structure")
        print("="*70 + "\n")

        base_dir = Path(__file__).parent.parent

        directories = [
            'data/raw',
            'data/derivatives',
            'results/preprocessing',
            'results/connectivity',
            'results/visualization',
            'results/qc',
            'results/reports',
            'logs',
        ]

        for dir_path in directories:
            full_path = base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {dir_path}")

        print("\n✓ Directory structure created!")

    def print_summary(self):
        """Print installation summary."""
        print("\n" + "="*70)
        print("Installation Summary")
        print("="*70 + "\n")

        if self.missing_software:
            print("⚠ Missing neuroimaging software (optional):")
            for software in self.missing_software:
                print(f"  - {software}")
            print("\nNote: These are pre-installed on your system according to requirements.")
            print("If needed, install them following their official documentation.")

        print("\n✓ Python environment is ready!")
        print("\nNext steps:")
        print("  1. Place your fMRI data in: data/raw/")
        print("     Format: data/raw/sub-{ID}/anat/ and data/raw/sub-{ID}/func/")
        print("  2. Review configuration: config/pipeline_config.yaml")
        print("  3. Run single subject: python pipelines/single_subject.py sub-001")
        print("  4. Run batch processing: python pipelines/batch_processing.py")

    def run_installation(self):
        """Run complete installation process."""
        print("\n" + "="*70)
        print("fMRI Preprocessing Pipeline - Dependency Installation")
        print("="*70)

        print(f"\nPython version: {sys.version}")
        print(f"Platform: {platform.platform()}")

        # Check software
        self.check_software_dependencies()

        # Install Python packages
        if not self.install_python_packages():
            print("\n✗ Installation failed!")
            return False

        # Verify imports
        if not self.verify_python_imports():
            print("\n✗ Package verification failed!")
            return False

        # Download atlases
        self.download_atlases()

        # Create directories
        self.create_directory_structure()

        # Print summary
        self.print_summary()

        return True


def main():
    """Main entry point."""
    installer = DependencyInstaller()
    success = installer.run_installation()

    if success:
        print("\n✓ Installation completed successfully!")
        return 0
    else:
        print("\n✗ Installation completed with errors.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
