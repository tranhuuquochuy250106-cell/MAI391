"""Independent numerical and input-integrity checks for the EDA handoff."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    res = ROOT / 'results'
    contract = json.loads((res / 'data_contract.json').read_text(encoding='utf-8'))
    train = pd.read_csv(ROOT / 'data/processed/train.csv')
    stats = pd.read_csv(res / 'q2_descriptive_stats.csv').set_index('variable')
    corr = pd.read_csv(res / 'q2_correlation.csv').iloc[0]
    bm = pd.read_csv(res / 'q2_binned_means.csv')
    iqr = pd.read_csv(res / 'q2_iqr_check.csv').set_index('variable')
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError('Q2 check failed: ' + name)

    check('train_only_descriptive_statistics', set(stats.dataset) == {'train'} and int(corr.n) == len(train))
    for rel, expected in contract['sha256'].items():
        check('unchanged_' + rel, hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected)
    for name in ('AT', 'PE'):
        x = train[name].to_numpy()
        row = stats.loc[name]
        values = {'n': len(x), 'mean': float(np.mean(x)), 'std_ddof1': float(np.std(x, ddof=1)),
                  'min': float(np.min(x)), 'Q1': float(np.quantile(x, .25)),
                  'median': float(np.median(x)), 'Q3': float(np.quantile(x, .75)), 'max': float(np.max(x))}
        check(name + '_statistics_match_numpy', all(np.isclose(float(row[k]), v, rtol=0, atol=1e-10) for k, v in values.items()))
        lo, hi = float(iqr.loc[name, 'lower_fence']), float(iqr.loc[name, 'upper_fence'])
        check(name + '_IQR_counts_match', int(((x < lo) | (x > hi)).sum()) == int(iqr.loc[name, 'n_outside_fence']))
    x, y = train.AT.to_numpy(), train.PE.to_numpy()
    xc, yc = x-x.mean(), y-y.mean()
    r = float(np.dot(xc, yc) / np.sqrt(np.dot(xc, xc)*np.dot(yc, yc)))
    check('Pearson_matches_centered_dot_product', abs(r - float(corr.pearson_r)) < 1e-12)
    check('binned_counts_cover_all_training_rows', int(bm.n.sum()) == len(train))
    check('weighted_bin_means_recover_train_means', np.isclose(np.average(bm.AT_mean, weights=bm.n), x.mean(), atol=1e-12, rtol=0) and np.isclose(np.average(bm.PE_mean, weights=bm.n), y.mean(), atol=1e-10, rtol=0))
    check('temperature_bins_cover_maximum', float(bm.bin_upper.max()) >= x.max())
    check('sparse_bins_are_flagged', np.array_equal(bm.trend_included.to_numpy(), (bm.n >= 30).to_numpy()))
    check('no_rows_removed_in_Q2', int(iqr.rows_removed.sum()) == 0)
    check('three_figures_created', all((ROOT / 'figures' / n).is_file() for n in ['q2_scatter.png', 'q2_distributions.png', 'q2_boxplots.png']))
    pd.DataFrame(checks).to_csv(res / 'q2_verification.csv', index=False)
    print(f'Q2 CHECK: PASS ({len(checks)} checks)')
    print('Training data unchanged; all temperature bins covered; no rows removed.')

if __name__ == '__main__':
    main()
