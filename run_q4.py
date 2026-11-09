"""Verify the shared data and execute Q4 training without preparing a new split."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    runpy.run_path(str(ROOT / 'src/verify_data.py'), run_name='__main__')
    from src.q4_training import main as run_training
    run_training()

if __name__ == '__main__':
    main()
