"""Check a downloaded upload package against a frozen size/SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / 'data/release_2.0.0-rc1/SHA256SUMS.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, manifest=DEFAULT_MANIFEST):
    root, manifest = Path(root), Path(manifest)
    expected = json.loads(manifest.read_text(encoding='utf-8-sig'))
    uploaded = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8-sig'))
    if uploaded != expected:
        raise ValueError('Downloaded checksum manifest differs from the expected frozen manifest')
    names = set()
    for item in expected:
        name = item['file']
        if Path(name).name != name or name in names:
            raise ValueError('Invalid or duplicate manifest filename: ' + name)
        names.add(name)
        path = root / name
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('Size/SHA-256 mismatch: ' + name)
        print('VERIFIED', name, flush=True)
    return {'status': 'passed', 'files': len(names) + 1,
            'doi_publication_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root, args.manifest), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
