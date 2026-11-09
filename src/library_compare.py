"""Q9 integration of TV2's three-method comparison on the locked split."""
import json
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from .analytical import to_raw_scale
from .data_io import ROOT, design_matrix, load_training_data
from .gradient_descent import gradient_descent
from .linear_model import compute_cost, predict
from .reference import metrics

N_REPEAT = 30


def time_fit(fit):
    """Warm up once, then time only fitting with data already in memory."""
    fit()
    times = []
    for _ in range(N_REPEAT):
        start = perf_counter()
        fit()
        times.append(perf_counter() - start)
    return {
        'fit_time_median_s': float(np.median(times)),
        'fit_time_min_s': min(times),
        'fit_time_max_s': max(times),
        'n_repeat': N_REPEAT,
    }


def main():
    X_train, y_train, train, scaling = load_training_data()
    results = ROOT / 'results'
    config = json.loads((results / 'baseline_config.json').read_text(encoding='utf-8'))
    contract = json.loads((results / 'data_contract.json').read_text(encoding='utf-8'))
    q5 = json.loads((results / 'q5_summary.json').read_text(encoding='utf-8'))
    expected_inputs = {
        'train_sha256': 'data/processed/train.csv',
        'test_sha256': 'data/processed/test.csv',
        'scaling_sha256': 'results/scaling.json',
    }
    if (config['is_mock_data'] or not config['converged']
            or config['data_version'] != contract['data_version']
            or config['data_version'] != q5['data_version']
            or any(config[key] != contract['sha256'][name]
                   for key, name in expected_inputs.items())):
        raise ValueError('Q4 and Q5 must match the verified data contract.')

    test = pd.read_csv(ROOT / 'data/processed/test.csv')
    if (len(test) != contract['n_test'] or not test.source_row.is_unique
            or not set(test.source_row).isdisjoint(train.source_row)):
        raise ValueError('Unexpected or overlapping test rows.')
    y_test = test.PE.to_numpy(dtype=float)
    X_test = design_matrix(test.AT.to_numpy(), scaling)
    np.testing.assert_allclose(X_test[:, 1], test.AT_scaled, atol=1e-12, rtol=0)
    z_train, z_test = X_train[:, 1:2], X_test[:, 1:2]
    with np.load(results / 'baseline_run.npz', allow_pickle=False) as baseline:
        gd = baseline['theta_final']
        np.testing.assert_array_equal(gd, baseline['theta_history'][-1])
        np.testing.assert_array_equal(baseline['source_row_train'], train.source_row)
        np.testing.assert_array_equal(baseline['source_row_test'], test.source_row)
        gd_train, gd_test = predict(X_train, gd), predict(X_test, gd)
        np.testing.assert_allclose(gd_train, baseline['pred_train'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(gd_test, baseline['pred_test'], atol=1e-9, rtol=0)

    def fit_gd():
        return gradient_descent(
            X_train, y_train, config['theta_initial'], learning_rate=config['learning_rate'],
            max_iter=config['max_iter'], tolerance=config['tolerance'],
        )

    def fit_analytical():
        return np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)

    def fit_library():
        return LinearRegression(fit_intercept=True).fit(z_train, y_train)

    with threadpool_limits(limits=1):
        replay = fit_gd()
        if replay['status'] != config['status'] or replay['n_iter'] != config['n_iter']:
            raise ValueError('Timed GD configuration does not reproduce the baseline.')
        np.testing.assert_allclose(replay['theta_final'], gd, atol=1e-9, rtol=0)
        analytical = fit_analytical()
        np.testing.assert_allclose(analytical, q5['theta_scaled'], atol=1e-9, rtol=0)
        library = fit_library()
        library_theta = np.array([library.intercept_, library.coef_[0]])
        timings = [time_fit(fit) for fit in (fit_gd, fit_analytical, fit_library)]

    ref_train, ref_test = predict(X_train, analytical), predict(X_test, analytical)
    lib_train, lib_test = library.predict(z_train), library.predict(z_test)
    np.testing.assert_allclose(lib_train, predict(X_train, library_theta), atol=1e-9, rtol=0)
    np.testing.assert_allclose(lib_test, predict(X_test, library_theta), atol=1e-9, rtol=0)
    rows = []
    fits = [
        ('gradient_descent', 'Q4 locked baseline', gd, gd_train, gd_test),
        ('analytical', 'Normal equations verified against Q5', analytical, ref_train, ref_test),
        ('sklearn', f'LinearRegression {sklearn.__version__}', library_theta, lib_train, lib_test),
    ]
    for (method, source, theta, pred_train, pred_test), timing in zip(fits, timings):
        intercept, slope = to_raw_scale(theta, scaling)
        train_scores = metrics(y_train, pred_train)
        test_scores = metrics(y_test, pred_test)
        j_train = compute_cost(X_train, y_train, theta)
        np.testing.assert_allclose(train_scores['MSE'], 2 * j_train, atol=1e-10, rtol=0)
        tolerance = q5['max_prediction_difference_tolerance_MW'] if method == 'gradient_descent' else 1e-9
        np.testing.assert_allclose(pred_train, ref_train, atol=tolerance, rtol=0)
        np.testing.assert_allclose(pred_test, ref_test, atol=tolerance, rtol=0)
        rows.append({
            'method': method, 'source': source, 'intercept_scaled': theta[0],
            'slope_scaled': theta[1], 'intercept_raw_MW': intercept,
            'slope_raw_MW_per_C': slope, 'J_train': j_train,
            'MSE_train': train_scores['MSE'], 'MSE_test': test_scores['MSE'],
            'R2_test': test_scores['R2'],
            'max_abs_pred_diff_train_vs_analytical_MW': float(np.max(np.abs(pred_train - ref_train))),
            'max_abs_pred_diff_test_vs_analytical_MW': float(np.max(np.abs(pred_test - ref_test))),
            **timing,
        })

    q8 = pd.read_csv(results / 'q8_metrics.csv').set_index('split')
    np.testing.assert_allclose(
        [rows[0]['MSE_train'], rows[0]['MSE_test'], rows[0]['R2_test']],
        [q8.at['train', 'MSE_MW2'], q8.at['test', 'MSE_MW2'], q8.at['test', 'R2']],
        atol=1e-10, rtol=0,
    )
    pd.DataFrame(rows).to_csv(results / 'q9_comparison.csv', index=False)
    summary = {
        'data_version': config['data_version'], 'n_train': len(train), 'n_test': len(test),
        'gd_iterations': replay['n_iter'], 'warmup_per_method': 1, 'timed_fits_per_method': N_REPEAT,
        'numerical_thread_limit': 1, 'python': platform.python_version(),
        'platform': platform.platform(), 'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
        'verification': 'PASS',
    }
    (results / 'q9_summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                            encoding='utf-8')
    print(f'Q9 CHECK: PASS (3 methods, {N_REPEAT} timed fits per method)')


if __name__ == '__main__':
    main()
