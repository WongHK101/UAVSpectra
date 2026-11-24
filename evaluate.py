"""Convenience entry point for GCP, LiDAR and proxy evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_command(args):
    if args.metric == 'proxy':
        return [sys.executable, '-m', 'portable_proxy', '--scene', args.scene,
                '--reference', str((args.data / 'proxy' / args.scene).resolve()),
                '--predictions', str(args.predictions.resolve()), '--output', str(args.output.resolve())]
    configuration = json.loads((ROOT / 'evaluation/configs/scoring.json').read_text())
    scene = configuration['scenes'][args.scene]
    command = [sys.executable, '-m', 'tools.measured_geometry_v1.score_native_' + args.metric]
    keys = ('sidecar', 'geometry', 'targets') if args.metric == 'gcp' else ('sidecar', 'geometry', 'surface_list', 'binding')
    for key in keys:
        record = scene[key]
        flag = key.replace('_', '-')
        command += ['--' + flag, str((args.data / record['path']).resolve()),
                    '--' + flag + '-sha256', record['sha256']]
    command += ['--reference-root', str(ROOT / 'evaluation/third_party/reference'),
                '--reference-manifest-sha256', configuration['reference_manifest_sha256'],
                '--output', str(args.output.resolve())]
    if args.metric == 'gcp':
        command += ['--export-root', str(args.export_root.resolve()),
                    '--export-manifest-sha256', file_hash(args.export_root / 'export_manifest.json')]
    else:
        command += ['--exports', str(args.exports.resolve()), '--exports-sha256', file_hash(args.exports),
                    '--lidar-root', str((args.data / 'lidar').resolve())]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=('gcp', 'lidar', 'proxy'), required=True)
    parser.add_argument('--scene', choices=('road', 'five_k', 'eucalyptus', 'maize', 'cassava', 'papaya'), required=True)
    parser.add_argument('--data', type=Path, required=True, help='UAVSpectra-dataset directory')
    parser.add_argument('--export-root', type=Path, help='Method moment-packet export, for GCP')
    parser.add_argument('--exports', type=Path, help='Method export-list JSON, for LiDAR')
    parser.add_argument('--predictions', type=Path, help='Common-camera packet manifest, for proxy')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.metric == 'gcp' and not args.export_root) or (args.metric == 'lidar' and not args.exports):
        parser.error('GCP requires --export-root; LiDAR requires --exports')
    if args.metric == 'proxy' and not args.predictions:
        parser.error('Proxy requires --predictions')
    if args.metric != 'proxy' and args.scene not in ('road', 'five_k'):
        parser.error('GCP and LiDAR are available for road and five_k')
    if args.output.exists():
        parser.error('Output directory already exists')
    subprocess.run(build_command(args), cwd=ROOT / 'evaluation', check=True)


if __name__ == '__main__':
    main()
