"""Q3 cost/gradient evidence; no optimizer and no model-selection on test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data_io import load_training_data, verify_shared_files
from .linear_model import predict, compute_cost, compute_gradient

ROOT = Path(__file__).resolve().parents[1]

def finite_difference(X, y, theta, epsilon):
    basis = np.eye(len(theta))
    return np.array([(compute_cost(X, y, theta+epsilon*e) -
                      compute_cost(X, y, theta-epsilon*e))/(2*epsilon)
                     for e in basis])

def main():
    checks = []
    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError('Q3 verification failed: ' + name)

    A = np.array([[1., -1.], [1., 1.]])
    b = np.array([1., 3.])
    zero = np.array([0., 0.])
    check('hand_predictions_at_zero', np.array_equal(predict(A, zero), [0., 0.]))
    check('hand_SSE_10_and_J_2point5', np.sum((predict(A, zero)-b)**2) == 10 and compute_cost(A, b, zero) == 2.5)
    check('hand_gradient_minus2_minus1', np.array_equal(compute_gradient(A, b, zero), [-2., -1.]))
    perfect = np.array([2., 1.])
    check('perfect_predictions', np.array_equal(predict(A, perfect), b))
    check('zero_cost_at_perfect_predictions', compute_cost(A, b, perfect) == 0.)
    check('zero_gradient_at_perfect_predictions', np.array_equal(compute_gradient(A, b, perfect), [0., 0.]))
    evidence = []
    for theta in [zero, np.array([.7, -.3]), np.array([-2., 4.]), perfect]:
        exact = compute_gradient(A, b, theta)
        numeric = finite_difference(A, b, theta, 1e-5)
        err = float(np.max(np.abs(exact-numeric)))
        check('toy_finite_difference_' + str(theta.tolist()), err <= 1e-8)
        evidence.append({'dataset': 'two_point_hand_example', 'theta0': theta[0], 'theta1': theta[1],
                         'epsilon': 1e-5, 'J': compute_cost(A, b, theta),
                         'gradient0': exact[0], 'gradient1': exact[1],
                         'finite_difference0': numeric[0], 'finite_difference1': numeric[1],
                         'max_abs_error': err, 'absolute_tolerance': 1e-8, 'passed': err <= 1e-8})

    X, y, train, scaling = load_training_data()
    check('shared_training_shapes', X.shape == (7622, 2) and y.shape == (7622,) and zero.shape == (2,))
    check('intercept_column_is_one', np.array_equal(X[:, 0], np.ones(len(y))))
    check('AT_only_standardized_feature', abs(X[:, 1].mean()) < 1e-12 and abs(X[:, 1].std(ddof=0)-1.) < 1e-12)
    yhat = predict(X, zero)
    error = yhat-y
    sse_loop = sum(float(e)*float(e) for e in error)
    J0 = compute_cost(X, y, zero)
    g0 = compute_gradient(X, y, zero)
    g_loop = np.array([sum(float(e) for e in error)/len(y),
                       sum(float(e)*float(z) for e,z in zip(error, X[:, 1]))/len(y)])
    check('train_cost_matches_scalar_sum', np.isclose(J0, sse_loop/(2*len(y)), rtol=0, atol=1e-8))
    check('train_gradient_matches_scalar_sums', np.allclose(g0, g_loop, rtol=0, atol=1e-9))
    check('train_MSE_equals_2J', np.isclose(np.mean(error**2), 2*J0, rtol=0, atol=1e-8))
    for theta in [zero, np.array([450., -10.]), np.array([470., 5.])]:
        exact = compute_gradient(X, y, theta)
        numeric = finite_difference(X, y, theta, 1e-4)
        err = float(np.max(np.abs(exact-numeric)))
        check('train_finite_difference_' + str(theta.tolist()), err <= 1e-6)
        evidence.append({'dataset': 'shared_train', 'theta0': theta[0], 'theta1': theta[1],
                         'epsilon': 1e-4, 'J': compute_cost(X, y, theta),
                         'gradient0': exact[0], 'gradient1': exact[1],
                         'finite_difference0': numeric[0], 'finite_difference1': numeric[1],
                         'max_abs_error': err, 'absolute_tolerance': 1e-6, 'passed': err <= 1e-6})

    bad_cases = [(X[:2], y[:2, None], zero), (X[:2], y[:2], [0.]),
                 (np.empty((0, 2)), np.array([]), zero),
                 (A, np.array([np.nan, 3.]), zero)]
    for i, (badX, bady, badtheta) in enumerate(bad_cases, 1):
        rejected = False
        try:
            compute_cost(badX, bady, badtheta)
        except ValueError:
            rejected = True
        check('invalid_input_rejected_' + str(i), rejected)
    contract = verify_shared_files()
    check('shared_inputs_unchanged', True)

    res = ROOT / 'results'
    pd.DataFrame(evidence).to_csv(res / 'q3_gradient_checks.csv', index=False)
    pd.DataFrame(checks).to_csv(res / 'q3_verification.csv', index=False)
    pd.DataFrame([
        {'dataset': 'two_point_hand_example', 'theta0': 0., 'theta1': 0., 'n': 2,
         'SSE': 10., 'J_half_MSE': 2.5, 'MSE': 5., 'gradient0': -2., 'gradient1': -1.},
        {'dataset': 'two_point_hand_example_perfect', 'theta0': 2., 'theta1': 1., 'n': 2,
         'SSE': 0., 'J_half_MSE': 0., 'MSE': 0., 'gradient0': 0., 'gradient1': 0.},
        {'dataset': 'shared_train_initialization_only', 'theta0': 0., 'theta1': 0., 'n': len(y),
         'SSE': float(error @ error), 'J_half_MSE': J0, 'MSE': float(np.mean(error**2)),
         'gradient0': g0[0], 'gradient1': g0[1]}
    ]).to_csv(res / 'q3_cost_examples.csv', index=False)
    summary = {'question': 3, 'data_version': contract['data_version'], 'n_train': len(y),
               'X_shape': list(X.shape), 'y_shape': list(y.shape), 'theta_shape': [2],
               'theta_evaluation': zero.tolist(), 'J_at_initial_theta': J0,
               'MSE_at_initial_theta': float(np.mean(error**2)), 'gradient_at_initial_theta': g0.tolist(),
               'scaling': {'mean_AT': scaling['mean_AT'], 'std_AT': scaling['std_AT'], 'ddof': scaling['ddof']},
               'finite_difference_checks': len(evidence),
               'max_abs_error_toy': max(e['max_abs_error'] for e in evidence if e['dataset'] == 'two_point_hand_example'),
               'max_abs_error_train': max(e['max_abs_error'] for e in evidence if e['dataset'] == 'shared_train'),
               'checks_passed': len(checks), 'optimizer_run': False, 'test_metrics_computed': False,
               'input_sha256': contract['sha256'],
               'source': 'Core numerical functions copied unchanged from submitted TV3 linear_model.py.',
               'scope': 'Cost and gradient verification only. Initial cost is not an optimized-model result.'}
    (res / 'q3_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('X shape:', X.shape, '| y shape:', y.shape, '| theta shape:', zero.shape)
    print(f'Toy example: J=2.5, MSE=5.0, gradient=[-2.0, -1.0]')
    print(f'Train at theta=[0,0]: J={J0:.12f}; MSE={np.mean(error**2):.12f}')
    print('Gradient at theta=[0,0]:', g0.tolist())
    print(f'Q3 CHECK: PASS ({len(checks)} checks; {len(evidence)} finite-difference points)')
    print('No Gradient Descent run; no optimized parameters; shared data unchanged.')

if __name__ == '__main__':
    main()
