"""Build and verify the shared data for MAI391 question 1."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    runpy.run_path(str(ROOT / 'src/data_prep.py'), run_name='__main__')
    runpy.run_path(str(ROOT / 'src/verify_data.py'), run_name='__main__')

if __name__ == '__main__':
    main()
