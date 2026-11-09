"""Read the shared Q1 training data without re-splitting or changing scaling."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def verify_shared_files():
    contract = json.loads((ROOT / 'results/data_contract.json').read_text(encoding='utf-8'))
    for rel, expected in contract['sha256'].items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Shared input changed: ' + rel)
    return contract

def design_matrix(at_raw, scaling):
    """Make [1, z] from a one-dimensional array of temperatures."""
    at = np.asarray(at_raw, dtype=float)
    mu, sd = float(scaling['mean_AT']), float(scaling['std_AT'])
    if at.ndim != 1 or len(at) == 0 or not np.isfinite(at).all():
        raise ValueError('AT must be a nonempty finite one-dimensional array.')
    if not np.isfinite([mu, sd]).all() or sd <= 0:
        raise ValueError('Saved train scaling must be finite with positive std.')
    return np.column_stack([np.ones(len(at)), (at-mu)/sd])

def load_training_data():
    contract = verify_shared_files()
    train = pd.read_csv(ROOT / 'data/processed/train.csv')
    scaling = json.loads((ROOT / 'results/scaling.json').read_text(encoding='utf-8'))
    if len(train) != contract['n_train'] or not train.source_row.is_unique:
        raise ValueError('Unexpected train rows or duplicate source_row.')
    X = design_matrix(train.AT.to_numpy(), scaling)
    y = train.PE.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Target PE must be finite.')
    np.testing.assert_allclose(X[:, 1], train.AT_scaled.to_numpy(), atol=1e-12, rtol=0)
    return X, y, train, scaling
