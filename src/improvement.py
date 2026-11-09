"""Choose a polynomial degree inside train, then compare on the locked test."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_io import ROOT, design_matrix, load_training_data
from .gradient_descent import gradient_descent
from .linear_model import predict
from .reference import metrics

SEED = 42
VALIDATION_FRACTION = 0.2


def fit_design(at, degree):
    scaling = {'mean_AT': float(at.mean()), 'std_AT': float(at.std(ddof=0))}
    if degree == 2:
        z_squared = design_matrix(at, scaling)[:, 1] ** 2
        scaling.update(mean_z_squared=float(z_squared.mean()),
                       std_z_squared=float(z_squared.std(ddof=0)))
    return scaling


def polynomial_design(at, scaling):
    X = design_matrix(at, scaling)
    if 'mean_z_squared' in scaling:
        squared = (X[:, 1] ** 2 - scaling['mean_z_squared']) / scaling['std_z_squared']
        X = np.column_stack((X, squared))
    return X


def fit_candidate(at, y, degree, config):
    scaling = fit_design(at, degree)
    X = polynomial_design(at, scaling)
    run = gradient_descent(
        X, y, np.zeros(X.shape[1]), learning_rate=config['learning_rate'],
        max_iter=config['max_iter'], tolerance=config['tolerance'],
    )
    if not run['converged'] or run['gradient_norm'] > config['tolerance']:
        raise ValueError(f'Degree {degree} failed the GD stopping criterion.')
    return run, scaling


def plot_comparison(train_at, train_y, test_at, predictions, output):
    fig, (fit_ax, resid_ax) = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    fit_ax.scatter(train_at, train_y, s=8, color='#748b9b', alpha=0.25,
                   edgecolors='none', label='Quan sát train')
    # Use train-defined temperature groups for the diagnostic curves.
    cuts = np.quantile(train_at, np.arange(1, 10) / 10)
    edges = np.r_[-np.inf, cuts, np.inf]
    for label, color, curve, train_resid, test_resid in predictions:
        fit_ax.plot(*curve, linewidth=2.1, color=color, label=label)
        resid_ax.scatter(test_at, test_resid, s=8, color=color, alpha=0.15,
                         edgecolors='none')
        groups = pd.DataFrame({'AT': train_at, 'e': train_resid})
        bins = groups.groupby(pd.cut(train_at, edges), observed=True).mean()
        resid_ax.plot(bins.AT, bins.e, color=color, marker='o', markersize=3.5,
                      linewidth=1.8, label=label)
    resid_ax.axhline(0, color='#52616b', linewidth=1, linestyle='--')
    fit_ax.set(title='Đường hồi quy trên dữ liệu train', ylabel='PE (MW)')
    resid_ax.set(title='Phần dư test và trung bình theo nhóm train', ylabel='PE thực − PE dự đoán (MW)')
    for ax in (fit_ax, resid_ax):
        ax.set_xlabel('AT (°C)')
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle('Câu 10 · So sánh mô hình cơ sở và phương án được chọn',
                 fontsize=14, fontweight='bold')
    fig.savefig(output, dpi=210)
    plt.close(fig)


def main():
    X_train, y_train, train, locked_scaling = load_training_data()
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
        raise ValueError('Q4 baseline does not match the verified data.')

    at = train.AT.to_numpy(dtype=float)
    order = np.random.default_rng(SEED).permutation(len(train))
    n_validation = round(VALIDATION_FRACTION * len(train))
    validation, fitting = order[:n_validation], order[n_validation:]
    selection = []
    for degree in (1, 2):
        run, scaling = fit_candidate(at[fitting], y_train[fitting], degree, config)
        predicted = predict(polynomial_design(at[validation], scaling), run['theta_final'])
        score = metrics(y_train[validation], predicted)
        selection.append({'degree': degree, 'n_fit': len(fitting), 'n_validation': len(validation),
                          'n_iter': run['n_iter'], **score})
    # Selection uses validation only, with the simpler model winning exact ties.
    selected = min(selection, key=lambda row: (row['MSE'], row['degree']))['degree']
    run, scaling = fit_candidate(at, y_train, selected, config)
    theta = run['theta_final']
    expanded_train = predict(polynomial_design(at, scaling), theta)

    test = pd.read_csv(ROOT / 'data/processed/test.csv')
    if (len(test) != contract['n_test'] or not test.source_row.is_unique
            or not set(test.source_row).isdisjoint(train.source_row)):
        raise ValueError('Unexpected or overlapping test rows.')
    test_at = test.AT.to_numpy(dtype=float)
    y_test = test.PE.to_numpy(dtype=float)
    X_test = design_matrix(test_at, locked_scaling)
    expanded_test = predict(polynomial_design(test_at, scaling), theta)

    with np.load(results / 'baseline_run.npz', allow_pickle=False) as baseline:
        gd = baseline['theta_final']
        np.testing.assert_array_equal(gd, baseline['theta_history'][-1])
        np.testing.assert_array_equal(train.source_row, baseline['source_row_train'])
        np.testing.assert_array_equal(test.source_row, baseline['source_row_test'])
        baseline_train, baseline_test = predict(X_train, gd), predict(X_test, gd)
        np.testing.assert_allclose(baseline_train, baseline['pred_train'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(baseline_test, baseline['pred_test'], atol=1e-9, rtol=0)

    rows = []
    for method, degree, train_pred, test_pred in (
        ('baseline_gd', 1, baseline_train, baseline_test),
        ('validation_selected_gd', selected, expanded_train, expanded_test),
    ):
        for split, actual, predicted in (
            ('train', y_train, train_pred), ('test', y_test, test_pred),
        ):
            rows.append({'method': method, 'degree': degree, 'split': split,
                         'n': len(actual), **metrics(actual, predicted),
                         'mean_residual_MW': float(np.mean(actual - predicted))})
    q8 = pd.read_csv(results / 'q8_metrics.csv').set_index('split')
    for row in rows[:2]:
        np.testing.assert_allclose(
            [row['MSE'], row['RMSE'], row['R2']],
            q8.loc[row['split'], ['MSE_MW2', 'RMSE_MW', 'R2']].to_numpy(float),
            atol=1e-10, rtol=0,
        )

    pd.DataFrame(selection).assign(selected=lambda df: df.degree == selected).to_csv(
        results / 'q10_validation.csv', index=False
    )
    pd.DataFrame(rows).to_csv(results / 'q10_comparison.csv', index=False)
    pd.DataFrame({
        'source_row': test.source_row, 'AT_degC': test_at, 'PE_actual_MW': y_test,
        'PE_baseline_MW': baseline_test, 'PE_selected_MW': expanded_test,
        'residual_baseline_MW': y_test - baseline_test,
        'residual_selected_MW': y_test - expanded_test,
    }).to_csv(results / 'q10_test_predictions.csv', index=False)
    grid = np.linspace(at.min(), at.max(), 200)
    plot_comparison(at, y_train, test_at, [
        ('Baseline bậc một', '#176985',
         (grid, predict(design_matrix(grid, locked_scaling), gd)),
         y_train - baseline_train, y_test - baseline_test),
        (f'Mô hình chọn · bậc {selected}', '#a44d22',
         (grid, predict(polynomial_design(grid, scaling), theta)),
         y_train - expanded_train, y_test - expanded_test),
    ], ROOT / 'figures/q10_comparison.png')

    summary = {
        'data_version': config['data_version'], 'selection_metric': 'validation MSE',
        'validation_seed': SEED, 'validation_fraction': VALIDATION_FRACTION,
        'selected_degree': selected, 'full_train_scaling': scaling, 'theta': theta.tolist(),
        'full_train_iterations': run['n_iter'], 'full_train_gradient_norm': run['gradient_norm'],
        'test_MSE_reduction_percent': 100 * (1 - rows[3]['MSE'] / rows[1]['MSE']),
        'verification': 'PASS',
    }
    (results / 'q10_summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                             encoding='utf-8')
    print(f'Q10 CHECK: PASS (degree {selected} selected on validation)')


if __name__ == '__main__':
    main()
