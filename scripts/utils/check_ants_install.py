#!/usr/bin/env python3
"""
Utility to diagnose ANTs installation issues.

The script reports:
  * ANTSPATH/PATH values
  * Locations of antsRegistration, antsApplyTransforms, antsRegistrationSyN.sh
  * Version output for the binaries (if available)
  * Optional end-to-end antsRegistrationSyN.sh test run on user-provided images
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, object]:
    """Run a command and capture stdout/stderr."""
    record: Dict[str, object] = {
        'cmd': cmd,
    }
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        record.update({
            'status': 'ok',
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        })
    except FileNotFoundError as exc:
        record.update({
            'status': 'missing',
            'error': str(exc),
        })
    except subprocess.CalledProcessError as exc:
        record.update({
            'status': 'error',
            'returncode': exc.returncode,
            'stdout': exc.stdout,
            'stderr': exc.stderr,
            'error': str(exc),
        })
    return record


def gather_binary_info() -> Dict[str, object]:
    """Collect PATH/ANTSPATH and executable locations."""
    binaries = ['antsRegistration', 'antsApplyTransforms', 'antsRegistrationSyN.sh']
    data = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'ANTSPATH': os.environ.get('ANTSPATH'),
        'PATH': os.environ.get('PATH'),
        'binaries': {},
    }
    for name in binaries:
        data['binaries'][name] = shutil.which(name)
    return data


def perform_test_run(args: argparse.Namespace) -> Dict[str, object]:
    """Execute antsRegistrationSyN.sh with supplied inputs."""
    out_prefix = Path(args.output_prefix).expanduser()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsRegistrationSyN.sh',
        '-d', str(args.dimensionality),
        '-f', args.fixed,
        '-m', args.moving,
        '-o', str(out_prefix),
        '-t', args.transform,
    ]
    if args.histogram_matching is not None:
        cmd.extend(['-j', str(args.histogram_matching)])
    if args.threads is not None:
        cmd.extend(['-n', str(args.threads)])

    result = run_command(cmd)

    expected = [
        f'{out_prefix}Warped.nii.gz',
        f'{out_prefix}InverseWarped.nii.gz',
        f'{out_prefix}0GenericAffine.mat',
    ]
    outputs = []
    for path in expected:
        path_obj = Path(path)
        outputs.append({'path': str(path_obj), 'exists': path_obj.exists()})

    result['expected_outputs'] = outputs

    if not args.keep_outputs:
        for entry in outputs:
            if entry['exists']:
                try:
                    Path(entry['path']).unlink()
                except OSError:
                    pass

    return result


def print_summary(info: Dict[str, object], version_checks: Dict[str, Dict[str, object]],
                  test_run: Optional[Dict[str, object]]) -> None:
    """Pretty print summary to stdout."""
    print("=== ANTs Diagnostic Report ===")
    print(f"Timestamp: {info['timestamp']}")
    print(f"ANTSPATH: {info['ANTSPATH']}")
    print("Binaries:")
    for name, path in info['binaries'].items():
        status = path if path else 'NOT FOUND'
        print(f"  - {name}: {status}")

    if version_checks:
        print("\nVersion Checks:")
        for name, res in version_checks.items():
            print(f"  [{name}] status={res.get('status')}")
            if res.get('stdout'):
                print(f"    stdout: {res['stdout'].strip()}")
            if res.get('stderr'):
                print(f"    stderr: {res['stderr'].strip()}")
            if res.get('error'):
                print(f"    error: {res['error']}")

    if test_run:
        print("\nTest Run:")
        print(f"  status={test_run.get('status')}")
        if test_run.get('error'):
            print(f"  error: {test_run['error']}")
        if test_run.get('stdout'):
            print("  stdout snippet:")
            print(indent_text(test_run['stdout']))
        if test_run.get('stderr'):
            print("  stderr snippet:")
            print(indent_text(test_run['stderr']))
        print("  Expected outputs:")
        for entry in test_run.get('expected_outputs', []):
            print(f"    - {entry['path']}: {'FOUND' if entry['exists'] else 'missing'}")
    print("=============================")


def indent_text(text: str, indent: str = "    ") -> str:
    """Indent multi-line text for readability."""
    return '\n'.join(f"{indent}{line}" for line in text.strip().splitlines())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose ANTs installation and optionally run a test registration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--json-report', help='Optional path to store JSON report.')
    parser.add_argument('--skip-versions', action='store_true',
                        help='Skip running --version checks.')
    parser.add_argument('--fixed', help='Fixed image for optional antsRegistrationSyN.sh test.')
    parser.add_argument('--moving', help='Moving image for optional antsRegistrationSyN.sh test.')
    parser.add_argument('--output-prefix', default='ants_diagnostic_',
                        help='Output prefix for antsRegistrationSyN.sh test.')
    parser.add_argument('--dimensionality', type=int, default=3,
                        help='Dimensionality for antsRegistrationSyN.sh test.')
    parser.add_argument('--transform', default='s',
                        help='Transform type for antsRegistrationSyN.sh.')
    parser.add_argument('--threads', type=int,
                        help='Number of threads to pass to antsRegistrationSyN.sh (-n).')
    parser.add_argument('--histogram-matching', type=int, choices=[0, 1],
                        help='Override histogram matching flag (-j).')
    parser.add_argument('--keep-outputs', action='store_true',
                        help='Do not delete test run outputs.')

    args = parser.parse_args()

    info = gather_binary_info()

    versions: Dict[str, Dict[str, object]] = {}
    if not args.skip_versions:
        versions['antsRegistration'] = run_command(['antsRegistration', '--version'])
        versions['antsApplyTransforms'] = run_command(['antsApplyTransforms', '--version'])
        versions['antsRegistrationSyN.sh'] = run_command(['antsRegistrationSyN.sh', '-h'])

    test_run = None
    if args.fixed and args.moving:
        test_run = perform_test_run(args)

    if args.json_report:
        report = {
            'environment': info,
            'versions': versions,
            'test_run': test_run,
        }
        Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_report, 'w') as f:
            json.dump(report, f, indent=2)

    print_summary(info, versions, test_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
