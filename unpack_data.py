"""Extract any downloaded UAVSpectra archives into the expected data layout."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile

SCENES = ('road', 'five_k', 'eucalyptus', 'maize', 'cassava', 'papaya')


def extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in item.filename or ':' in item.filename:
                raise ValueError('Unsafe archive member: ' + item.filename)
            target = destination.joinpath(*name.parts)
            if destination.resolve() not in target.resolve().parents:
                raise ValueError('Archive path escapes destination')
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                # Never replace an existing dataset file.
                raise FileExistsError(target)
        for item in source.infolist():
            if not item.is_dir():
                target = destination.joinpath(*PurePosixPath(item.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    print(archive.name + ' -> ' + str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Directory containing the downloaded ZIP files')
    parser.add_argument('--kind', choices=('all', 'reference', 'raw', 'processed', 'proxy'), default='all')
    args = parser.parse_args()
    if args.kind in ('all', 'reference'):
        archive = args.data / 'reference_data.zip'
        if archive.exists():
            extract(archive, args.data)
    for kind in ('raw', 'processed', 'proxy'):
        if args.kind not in ('all', kind):
            continue
        for scene in SCENES:
            archive = args.data / (kind + '_' + scene + '.zip')
            if archive.exists():
                extract(archive, args.data / kind / scene)


if __name__ == '__main__':
    main()
