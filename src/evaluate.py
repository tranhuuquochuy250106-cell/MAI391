"""Evaluation of the locked Q4 model for Q8, with shared NumPy metrics."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytical import to_raw_scale
from .data_io import ROOT, design_matrix, load_training_data
from .linear_model import compute_cost, predict
from .reference import metrics

COLORS = {'train': '#176985', 'test': '#a44d22'}


def evaluate_split(split, frame, predictions):
    actual = frame.PE.to_numpy(dtype=float)
    scores = metrics(actual, predictions)
    residuals = actual - predictions
    rows = pd.DataFrame({
        'source_row': frame.source_row, 'split': split, 'AT_degC': frame.AT,
        'PE_actual_MW': actual, 'PE_predicted_MW': predictions, 'residual_MW': residuals,
    })
    return {
        'split': split, 'n_observations': len(actual), 'MSE_MW2': scores['MSE'],
        'RMSE_MW': scores['RMSE'], 'R2': scores['R2'],
        'residual_mean_MW': float(residuals.mean()),
        'residual_std_MW': float(residuals.std()),
    }, rows


def residual_groups(rows, edges):
    groups = pd.cut(rows.PE_predicted_MW, bins=edges, labels=False)
    return (rows.assign(bin=groups + 1).groupby(['split', 'bin'], sort=False)
            .agg(n=('residual_MW', 'size'),
                 mean_predicted_MW=('PE_predicted_MW', 'mean'),
                 mean_AT_degC=('AT_degC', 'mean'),
                 mean_residual_MW=('residual_MW', 'mean'),
                 residual_std_MW=('residual_MW', lambda values: values.std(ddof=0)),
                 min_residual_MW=('residual_MW', 'min'),
                 max_residual_MW=('residual_MW', 'max'))
            .sort_index().reset_index())


def plot_fit(rows, theta_raw, output):
    fig, ax = plt.subplots(figsize=(9, 5.3), layout='constrained')
    for split, data in rows.groupby('split', sort=False):
        ax.scatter(data.AT_degC, data.PE_actual_MW, s=9, alpha=0.28,
                   color=COLORS[split], edgecolors='none', label=f'{split} · {len(data):,}')
    at = np.sort(rows.AT_degC.unique())
    ax.plot(at, theta_raw[0] + theta_raw[1] * at, color='#192936',
            linewidth=2.5, label='Đường hồi quy GD')
    ax.set(xlabel='Nhiệt độ AT (°C)', ylabel='Công suất PE (MW)',
           title='Câu 8 · Đường hồi quy và công suất quan sát')
    ax.legend(frameon=False)
    ax.grid(alpha=0.16)
    fig.savefig(output, dpi=210)
    plt.close(fig)


def plot_residuals(split, rows, bins, xlim, ylim, output):
    fig, ax = plt.subplots(figsize=(9, 5.1), layout='constrained')
    ax.scatter(rows.PE_predicted_MW, rows.residual_MW, s=10, alpha=0.27,
               color=COLORS[split], edgecolors='none', label='Phần dư từng quan sát')
    ax.axhline(0, color='#53616d', linewidth=1.2, linestyle='--', label='Mốc 0 MW')
    ax.plot(bins.mean_predicted_MW, bins.mean_residual_MW, color='#192936',
            marker='o', markersize=4, linewidth=1.6, label='Trung bình theo nhóm')
    ax.set(xlabel='PE dự đoán (MW)', ylabel='Phần dư (PE thực − PE dự đoán), MW',
           title=f'Câu 8 · Phần dư trên {split} ({len(rows):,} quan sát)',
           xlim=xlim, ylim=ylim)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(alpha=0.16)
    fig.savefig(output, dpi=210)
    plt.close(fig)


def main():
    X_train, y_train, train, scaling = load_training_data()
    results = ROOT / 'results'
    config = json.loads((results / 'baseline_config.json').read_text(encoding='utf-8'))
    contract = json.loads((results / 'data_contract.json').read_text(encoding='utf-8'))
    expected_inputs = {
        'train_sha256': 'data/processed/train.csv',
        'test_sha256': 'data/processed/test.csv',
        'scaling_sha256': 'results/scaling.json',
    }
    if (config['is_mock_data'] or not config['converged']
            or config['data_version'] != contract['data_version']
            or any(config[key] != contract['sha256'][name]
                   for key, name in expected_inputs.items())):
        raise ValueError('Q4 baseline does not match the verified data contract.')

    test = pd.read_csv(ROOT / 'data/processed/test.csv')
    if (len(test) != contract['n_test'] or not test.source_row.is_unique
            or not set(test.source_row).isdisjoint(train.source_row)):
        raise ValueError('Unexpected or overlapping test rows.')
    X_test = design_matrix(test.AT.to_numpy(), scaling)
    np.testing.assert_allclose(X_test[:, 1], test.AT_scaled, atol=1e-12, rtol=0)

    scores, row_frames = [], []
    with np.load(results / 'baseline_run.npz', allow_pickle=False) as baseline:
        theta = baseline['theta_final']
        np.testing.assert_array_equal(theta, baseline['theta_history'][-1])
        j_train = compute_cost(X_train, y_train, theta)
        np.testing.assert_allclose(j_train, baseline['cost_history'][-1], atol=1e-10, rtol=0)
        for split, frame, design in (('train', train, X_train), ('test', test, X_test)):
            np.testing.assert_array_equal(frame.source_row, baseline[f'source_row_{split}'])
            pred = predict(design, theta)
            np.testing.assert_allclose(pred, baseline[f'pred_{split}'], atol=1e-9, rtol=0)
            score, rows = evaluate_split(split, frame, pred)
            scores.append(score)
            row_frames.append(rows)

    mse_gap = abs(scores[0]['MSE_MW2'] - 2 * j_train)
    np.testing.assert_allclose(mse_gap, 0, atol=1e-10, rtol=0)
    theta_raw = to_raw_scale(theta, scaling)
    predictions = pd.concat(row_frames, ignore_index=True)
    np.testing.assert_allclose(predictions.PE_predicted_MW,
                               theta_raw[0] + theta_raw[1] * predictions.AT_degC,
                               atol=1e-10, rtol=0)
    temperatures = np.array([10., 20., 30.])
    if temperatures.min() < train.AT.min() or temperatures.max() > train.AT.max():
        raise ValueError('Inference examples must lie inside the observed train range.')
    X_examples = design_matrix(temperatures, scaling)
    examples = pd.DataFrame({
        'AT_degC': temperatures,
        'AT_scaled': X_examples[:, 1],
        'PE_predicted_MW': predict(X_examples, theta),
    })
    cuts = np.quantile(row_frames[0].PE_predicted_MW, np.arange(1, 10) / 10)
    bins = residual_groups(predictions, np.r_[-np.inf, cuts, np.inf])
    np.testing.assert_array_equal(bins.groupby('split').n.sum().sort_index(),
                                  predictions.groupby('split').size().sort_index())

    pd.DataFrame(scores).to_csv(results / 'q8_metrics.csv', index=False)
    predictions.to_csv(results / 'q8_predictions.csv', index=False)
    examples.to_csv(results / 'q8_examples.csv', index=False)
    bins.to_csv(results / 'q8_residual_bins.csv', index=False)

    plot_fit(predictions, theta_raw, ROOT / 'figures/q8_fit.png')
    xlim = (predictions.PE_predicted_MW.min() - 2, predictions.PE_predicted_MW.max() + 2)
    limit = float(predictions.residual_MW.abs().max() + 2)
    for split, rows in zip(('train', 'test'), row_frames):
        plot_residuals(split, rows, bins[bins.split == split], xlim, (-limit, limit),
                       ROOT / f'figures/q8_residual_{split}.png')

    summary = {
        'data_version': config['data_version'], 'theta_scaled': theta.tolist(),
        'theta_raw': theta_raw.tolist(), 'J_train': j_train,
        'abs_diff_MSE_train_vs_2J': mse_gap, 'prediction_bin_cutpoints_MW': cuts.tolist(),
        'r2_constant_target': 'undefined; empty field in CSV',
        'verification': 'PASS',
    }
    (results / 'q8_summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                            encoding='utf-8')
    print(f'Q8 CHECK: PASS ({len(predictions)} predictions verified)')


if __name__ == '__main__':
    main()
