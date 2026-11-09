"""Compute the Q5 reference and compare it with the locked Q4 baseline."""
import json

import numpy as np
import pandas as pd

from src.analytical import solve_least_squares, to_raw_scale
from src.data_io import ROOT, design_matrix, load_training_data
from src.linear_model import compute_cost, compute_gradient, predict


def main():
    results = ROOT / 'results'
    X, y, train, scaling = load_training_data()
    contract = json.loads((results / 'data_contract.json').read_text(encoding='utf-8'))
    config = json.loads((results / 'baseline_config.json').read_text(encoding='utf-8'))
    baseline_hashes = {
        'train_sha256': 'data/processed/train.csv',
        'test_sha256': 'data/processed/test.csv',
        'scaling_sha256': 'results/scaling.json',
    }
    if (config['data_version'] != contract['data_version'] or config['is_mock_data']
            or not config['converged']
            or any(config[key] != contract['sha256'][path]
                   for key, path in baseline_hashes.items())):
        raise ValueError('Q4 baseline does not match the verified data contract.')

    test = pd.read_csv(ROOT / 'data/processed/test.csv', usecols=['source_row', 'AT'])
    X_test = design_matrix(test.AT.to_numpy(), scaling)
    theta, theta_raw, raw_covariance = solve_least_squares(
        X, y, train.AT.to_numpy(), scaling
    )
    with np.load(results / 'baseline_run.npz', allow_pickle=False) as run:
        gd = run['theta_final']
        np.testing.assert_array_equal(run['source_row_train'], train.source_row)
        np.testing.assert_array_equal(run['source_row_test'], test.source_row)
        np.testing.assert_array_equal(gd, run['theta_history'][-1])
        if len(run['cost_history']) != config['n_iter'] + 1:
            raise ValueError('Q4 cost history does not match the saved iteration count.')
        np.testing.assert_allclose(compute_cost(X, y, gd), run['cost_history'][-1],
                                   atol=1e-10, rtol=0)
        for design, key in ((X, 'pred_train'), (X_test, 'pred_test')):
            np.testing.assert_allclose(predict(design, gd), run[key], atol=1e-9, rtol=0)

    gradient_norm = np.linalg.norm(compute_gradient(X, y, theta))
    if gradient_norm > 1e-9:
        raise ValueError('Analytical solution fails the stationary-point check.')

    rows = []
    for method, parameters, raw in (
        ('analytical', theta, theta_raw),
        ('gradient_descent', gd, to_raw_scale(gd, scaling)),
    ):
        train_gap = np.max(np.abs(predict(X, parameters) - predict(X, theta)))
        test_gap = np.max(np.abs(predict(X_test, parameters) - predict(X_test, theta)))
        rows.append({
            'method': method,
            'intercept_scaled': parameters[0],
            'slope_scaled': parameters[1],
            'intercept_raw_MW': raw[0],
            'slope_raw_MW_per_C': raw[1],
            'J_train': compute_cost(X, y, parameters),
            'gradient_l2_train': np.linalg.norm(compute_gradient(X, y, parameters)),
            'max_abs_pred_diff_train_vs_analytical_MW': train_gap,
            'max_abs_pred_diff_test_vs_analytical_MW': test_gap,
        })

    gd_row = rows[1]
    prediction_tolerance = 0.001
    if max(gd_row['max_abs_pred_diff_train_vs_analytical_MW'],
           gd_row['max_abs_pred_diff_test_vs_analytical_MW']) > prediction_tolerance:
        raise ValueError('GD prediction difference exceeds the agreed 0.001 MW limit.')
    if gd_row['gradient_l2_train'] > config['tolerance']:
        raise ValueError('Saved GD solution fails its original stopping criterion.')

    pd.DataFrame(rows).to_csv(results / 'q5_solution.csv', index=False)
    summary = {
        'data_version': config['data_version'],
        'n_train': len(train),
        'n_test': len(test),
        'normal_matrix_per_sample': ((X.T @ X) / len(y)).tolist(),
        'normal_rhs_per_sample': ((X.T @ y) / len(y)).tolist(),
        'theta_scaled': theta.tolist(),
        'theta_raw': theta_raw.tolist(),
        'raw_covariance_solution': raw_covariance.tolist(),
        'gd_iterations': config['n_iter'],
        'gd_theta_minus_analytical': (gd - theta).tolist(),
        'gd_parameter_l2_difference': float(np.linalg.norm(gd - theta)),
        'absolute_J_train_difference': abs(gd_row['J_train'] - rows[0]['J_train']),
        'max_prediction_difference_tolerance_MW': prediction_tolerance,
        'verification': 'PASS',
    }
    (results / 'q5_summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                             encoding='utf-8')
    print('Q5 CHECK: PASS')


if __name__ == '__main__':
    main()
