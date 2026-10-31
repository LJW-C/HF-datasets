"""Public entry point for the corrected, verified-source dataset builder."""
from pathlib import Path
import runpy
import sys


def main():
    directory = Path(__file__).resolve().parent / 'revision_v2'
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / 'build_verified_dataset.py'), run_name='__main__')


if __name__ == '__main__':
    main()
