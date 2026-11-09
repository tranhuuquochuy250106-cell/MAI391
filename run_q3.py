"""Verify Q1 shared data, then verify Q3 cost and gradient functions."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    runpy.run_path(str(ROOT / 'src/verify_data.py'), run_name='__main__')
    from src.q3_checks import main as run_checks
    run_checks()

if __name__ == '__main__':
    main()
