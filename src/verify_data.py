"""Independent checks and fingerprints for the group's shared data handoff."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ['AT', 'V', 'AP', 'RH', 'PE']

def main():
    raw_file = ROOT / 'data/raw/Folds5x2_pp.xlsx'
    raw = pd.read_excel(raw_file, sheet_name='Sheet1')
    clean = pd.read_csv(ROOT / 'data/processed/clean.csv')
    train = pd.read_csv(ROOT / 'data/processed/train.csv')
    test = pd.read_csv(ROOT / 'data/processed/test.csv')
    sc = json.loads((ROOT / 'results/scaling.json').read_text(encoding='utf-8'))
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError('Data verification failed: ' + name)

    check('raw_columns_match', list(raw.columns) == COLS)
    check('raw_is_finite', np.isfinite(raw[COLS].to_numpy()).all())
    check('raw_row_count_9568', len(raw) == 9568)
    check('full_row_duplicates_41', int(raw.duplicated(COLS).sum()) == 41)
    check('clean_row_count_9527', len(clean) == 9527)
    check('train_row_count_7622', len(train) == 7622)
    check('test_row_count_1905', len(test) == 1905)
    check('minimum_500_clean', len(clean) >= 500)
    check('no_remaining_full_row_duplicates', not clean.duplicated(COLS).any())
    check('source_rows_are_unique', all(d.source_row.is_unique for d in (clean, train, test)))
    check('train_test_disjoint', set(train.source_row).isdisjoint(test.source_row))
    check('train_test_cover_clean', set(train.source_row) | set(test.source_row) == set(clean.source_row))

    expected = raw.copy()
    expected.insert(0, 'source_row', np.arange(2, len(raw) + 2))
    expected = expected.drop_duplicates(COLS, keep='first').reset_index(drop=True)
    pd.testing.assert_frame_equal(clean, expected, check_exact=False, atol=1e-12, rtol=0)
    checks.append({'check': 'clean_matches_original_except_declared_dedup', 'passed': True})

    # Reconstruct the exact TV2 split, including row order.
    perm = np.random.default_rng(42).permutation(len(clean))
    n_test = int(round(.2 * len(clean)))
    expected_test_rows = clean.iloc[np.sort(perm[:n_test])].source_row.tolist()
    expected_train_rows = clean.iloc[np.sort(perm[n_test:])].source_row.tolist()
    check('exact_TV2_split_and_order', train.source_row.tolist() == expected_train_rows and test.source_row.tolist() == expected_test_rows)
    mu, sd = float(train.AT.mean()), float(train.AT.std(ddof=0))
    check('train_standard_deviation_positive', np.isfinite(sd) and sd > 0)
    check('saved_train_scaling', abs(sc['mean_AT'] - mu) < 1e-12 and abs(sc['std_AT'] - sd) < 1e-12)
    check('saved_counts_match', sc['n_clean'] == len(clean) and sc['n_train'] == len(train) and sc['n_test'] == len(test))
    for label, frame in [('train', train), ('test', test)]:
        check(label + '_is_finite', np.isfinite(frame[COLS + ['AT_scaled']].to_numpy()).all())
        np.testing.assert_allclose(frame.AT_scaled, (frame.AT - mu) / sd, atol=1e-12, rtol=0)
        original_values = clean.set_index('source_row').loc[frame.source_row, COLS].to_numpy()
        np.testing.assert_allclose(frame[COLS].to_numpy(), original_values, atol=1e-12, rtol=0)
        checks.append({'check': label + '_values_and_scaling_match', 'passed': True})
    check('train_scaled_mean_zero', abs(float(train.AT_scaled.mean())) < 1e-12)
    check('train_scaled_std_one', abs(float(train.AT_scaled.std(ddof=0)) - 1) < 1e-12)

    paths = ['data/raw/Folds5x2_pp.xlsx', 'data/processed/clean.csv', 'data/processed/train.csv', 'data/processed/test.csv', 'results/scaling.json']
    contract = {
        'data_version': 'TV2_shared_v1',
        'source_dataset': 'Combined Cycle Power Plant',
        'source_url': 'https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant',
        'source_doi': '10.24432/C5002N',
        'sheet': 'Sheet1', 'feature': 'AT', 'target': 'PE',
        'n_raw': len(raw), 'n_clean': len(clean), 'n_train': len(train), 'n_test': len(test),
        'mean_AT': mu, 'std_AT': sd, 'ddof': 0, 'seed': 42,
        'split': 'default_rng(42); first round(0.2*n) permuted indices are test; source-row order preserved within each split',
        'deduplication': 'Keep first occurrence of identical five-column rows; retain other observations',
        'RH_above_100_raw': int((raw.RH > 100).sum()), 'RH_max_raw': float(raw.RH.max()),
        'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        'checks_passed': len(checks),
        'scope': 'Question 1 shared data only. No GD or model evaluation results included.'
    }
    (ROOT / 'results/data_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(checks).to_csv(ROOT / 'results/q1_verification.csv', index=False, encoding='utf-8')
    print('\nQ1 DATA CHECK: PASS')
    print(f'raw={len(raw)}, clean={len(clean)}, train={len(train)}, test={len(test)}')
    print(f'mean_AT={mu:.12f}; std_AT={sd:.12f}; ddof=0')
    print(f'Checks passed: {len(checks)}')
    print('Shared data version: TV2_shared_v1')

if __name__ == '__main__':
    main()
