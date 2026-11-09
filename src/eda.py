"""Question 2 EDA, adapted from TV2 for the shared MAI391 data.

Descriptive exploration uses the training split. No models are fitted here.
All temperature bins include the full training range; sparse bins are reported
but do not determine the displayed bin-mean trend.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
RES = ROOT / 'results'
LABELS = {'AT': 'Ambient temperature AT (°C)', 'PE': 'Electrical power output PE (MW)'}
MIN_BIN_COUNT = 30

def main():
    FIG.mkdir(exist_ok=True)
    contract = json.loads((RES / 'data_contract.json').read_text(encoding='utf-8'))
    for rel, expected in contract['sha256'].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Shared data changed: {rel}. Run and review question 1 before EDA.')
    train = pd.read_csv(ROOT / 'data/processed/train.csv')
    if len(train) != contract['n_train'] or not np.isfinite(train[['AT', 'PE']].to_numpy()).all():
        raise ValueError('Unexpected training data.')

    rows = []
    for name in ('AT', 'PE'):
        s = train[name]
        rows.append({'dataset': 'train', 'variable': name, 'n': len(s),
                     'mean': s.mean(), 'std_ddof1': s.std(ddof=1),
                     'min': s.min(), 'Q1': s.quantile(.25), 'median': s.median(),
                     'Q3': s.quantile(.75), 'max': s.max(),
                     'skewness': s.skew(), 'unit': 'degC' if name == 'AT' else 'MW'})
    stats = pd.DataFrame(rows)
    stats.to_csv(RES / 'q2_descriptive_stats.csv', index=False)
    r = float(np.corrcoef(train.AT, train.PE)[0, 1])
    pd.DataFrame([{'dataset': 'train', 'pair': 'AT-PE', 'pearson_r': r,
                   'pearson_r_squared': r*r, 'n': len(train)}]).to_csv(RES / 'q2_correlation.csv', index=False)

    iqr_rows = []
    for name in ('AT', 'PE'):
        q1, q3 = train[name].quantile([.25, .75])
        width = q3 - q1
        low, high = q1 - 1.5*width, q3 + 1.5*width
        iqr_rows.append({'dataset': 'train', 'variable': name, 'Q1': q1, 'Q3': q3,
                         'IQR': width, 'lower_fence': low, 'upper_fence': high,
                         'n_outside_fence': int(((train[name] < low) | (train[name] > high)).sum()),
                         'rows_removed': 0})
    iqr = pd.DataFrame(iqr_rows)
    iqr.to_csv(RES / 'q2_iqr_check.csv', index=False)

    # The original TV2 edges stopped at 36, excluding AT=37.11 from bin summaries.
    # Full coverage is required even when an extreme bin has very few points.
    width = 4.0
    lower = np.floor(train.AT.min()/width)*width
    upper = np.ceil(train.AT.max()/width)*width
    edges = np.arange(lower, upper + width*.5, width)
    bins = pd.cut(train.AT, edges, right=True, include_lowest=True)
    if bins.isna().any():
        raise ValueError('Temperature bin edges do not cover all training observations.')
    grouped = train.assign(temperature_bin=bins).groupby('temperature_bin', observed=True)
    bm = grouped.agg(AT_mean=('AT', 'mean'), PE_mean=('PE', 'mean'),
                     PE_std_ddof1=('PE', 'std'), n=('PE', 'size')).reset_index()
    bm['bin_lower'] = bm.temperature_bin.map(lambda x: x.left).astype(float)
    bm['bin_upper'] = bm.temperature_bin.map(lambda x: x.right).astype(float)
    bm['trend_included'] = bm.n >= MIN_BIN_COUNT
    bm['temperature_bin'] = bm.temperature_bin.astype(str)
    bm.to_csv(RES / 'q2_binned_means.csv', index=False)
    if int(bm.n.sum()) != len(train):
        raise ValueError('Bin counts do not sum to the training sample size.')
    trend = bm.loc[bm.trend_included]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#202020',
                         'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(train.AT, train.PE, s=8, alpha=.24, color='#24678C',
               edgecolors='none', label=f'Training observations (n={len(train):,})')
    ax.plot(trend.AT_mean, trend.PE_mean, '-o', color='#BB4D28', lw=2.1, ms=4.5,
            label=f'4°C bin means (bins with n ≥ {MIN_BIN_COUNT})')
    ax.set(xlabel=LABELS['AT'], ylabel=LABELS['PE'],
           title=f'AT and PE on the training set   |   Pearson r = {r:.4f}')
    ax.grid(alpha=.15)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'q2_scatter.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, name in zip(axes, ('AT', 'PE')):
        ax.hist(train[name], bins=40, color='#35759A', edgecolor='white', linewidth=.5)
        ax.axvline(train[name].median(), color='#BB4D28', lw=1.7, ls='--', label='Median')
        ax.set(xlabel=LABELS[name], ylabel='Number of observations', title=f'{name} distribution')
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f'Training distributions (n={len(train):,})', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG / 'q2_distributions.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8))
    for ax, name in zip(axes, ('AT', 'PE')):
        ax.boxplot(train[name], patch_artist=True,
                   boxprops={'facecolor': '#D6E5ED', 'edgecolor': '#24678C'},
                   medianprops={'color': '#BB4D28', 'linewidth': 2},
                   whiskerprops={'color': '#24678C'}, capprops={'color': '#24678C'})
        ax.set(ylabel=LABELS[name], title=f'{name} boxplot', xticks=[])
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Training data   |   Whiskers use the 1.5 × IQR rule', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG / 'q2_boxplots.png', dpi=200)
    plt.close(fig)

    meta = {'question': 2, 'data_version': contract['data_version'],
            'analysis_split': 'train', 'n_train': len(train),
            'descriptive_std_ddof': 1, 'scaling_std_ddof_from_q1': 0,
            'pearson_r': r, 'bin_width_degC': width,
            'bin_edges': edges.tolist(), 'n_binned': int(bm.n.sum()),
            'minimum_bin_n_for_plotted_trend': MIN_BIN_COUNT,
            'n_in_sparse_bins': int(bm.loc[~bm.trend_included, 'n'].sum()),
            'rows_removed_in_q2': 0, 'model_fitted': False,
            'shared_inputs_unchanged': True, 'input_sha256': contract['sha256'],
            'source': 'Adapted from TV2 EDA; corrected full-range binning and explicit reporting conventions.'}
    (RES / 'q2_provenance.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(stats.round(6).to_string(index=False))
    print(f'Pearson r (train) = {r:.10f}')
    print('Bin coverage:', int(bm.n.sum()), '/', len(train))
    print('Q2 EDA: COMPLETE')

if __name__ == '__main__':
    main()
