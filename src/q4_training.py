"""Train and verify the shared Q4 baseline with TV3's unchanged NumPy GD.

Learning-rate experiments use train only. Test predictions are exported only
after the baseline is fixed; no test performance metrics are computed here.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_io import load_training_data, design_matrix, verify_shared_files
from .linear_model import predict, compute_cost, compute_gradient
from .gradient_descent import gradient_descent
from .reference import analytical_reference

ROOT = Path(__file__).resolve().parents[1]
MAX_ITER = 10000
TOLERANCE = 1e-6
THETA_INITIAL = np.array([0., 0.])
BASELINE_RATE = .1
PREDICTION_TOLERANCE_MW = .001

def stop_description(result):
    if result['status'] == 'converged':
        return 'L2 norm of the full-train gradient <= tolerance'
    if result['status'] == 'max_iter':
        return 'Reached max_iter before satisfying the gradient-norm tolerance'
    if result['status'] == 'diverged':
        return 'Cost exceeded 1e12 * max(initial_cost, 1)'
    return 'Nonfinite gradient or candidate; last finite stored state retained'

def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    X, y, train, scaling = load_training_data()
    contract = verify_shared_files()
    checks = []
    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError('Q4 check failed: ' + name)

    # Independent hand example checks the simultaneous update, without using the
    # analytical solution to update any GD parameters.
    A = np.array([[1., -1.], [1., 1.]])
    b = np.array([1., 3.])
    caller_theta = np.array([0., 0.])
    one = gradient_descent(A, b, caller_theta, learning_rate=.1, max_iter=1, tolerance=0.)
    check('hand_simultaneous_first_update', np.allclose(one['theta_final'], [.2, .1], atol=1e-15, rtol=0))
    check('hand_cost_after_first_update', abs(one['cost_history'][-1]-2.025) < 1e-14)
    check('initial_theta_not_mutated', np.array_equal(caller_theta, [0., 0.]))
    check('hand_step_zero_in_history', np.array_equal(one['theta_history'][0], [0., 0.]) and one['cost_history'][0] == 2.5)
    check('max_iter_is_not_convergence', one['status'] == 'max_iter' and not one['converged'])
    optimal = gradient_descent(A, b, [2., 1.])
    check('initial_optimum_stops_at_zero', optimal['status'] == 'converged' and optimal['n_iter'] == 0)
    unstable = gradient_descent(A, b, [0., 0.], learning_rate=3., max_iter=1000, divergence_factor=1e6)
    check('unstable_run_reported_as_diverged', unstable['status'] == 'diverged' and not unstable['converged'])
    check('unstable_run_keeps_finite_paired_history', len(unstable['theta_history']) == len(unstable['cost_history']) and np.isfinite(unstable['theta_history']).all() and np.isfinite(unstable['cost_history']).all())

    reference = analytical_reference(X, y)
    reference_J = compute_cost(X, y, reference)
    profiles = [('baseline', BASELINE_RATE), ('small', .01), ('large', 1.85),
                ('divergent', 2.05), ('one_step', 1.)]
    runs = {}
    run_rows = []
    res, fig_dir = ROOT / 'results', ROOT / 'figures'
    fig_dir.mkdir(exist_ok=True)
    for run_id, rate in profiles:
        result = gradient_descent(X, y, THETA_INITIAL, learning_rate=rate,
                                  max_iter=MAX_ITER, tolerance=TOLERANCE)
        runs[run_id] = result
        run_rows.append({'run_id': run_id, 'learning_rate': rate,
                         'theta0_initial': 0., 'theta1_initial': 0.,
                         'max_iter': MAX_ITER, 'tolerance_gradient_l2': TOLERANCE,
                         'n_iter': result['n_iter'], 'status': result['status'],
                         'converged': result['converged'], 'gradient_norm_final': result['gradient_norm'],
                         'J_initial': result['cost_history'][0], 'J_final': result['cost_history'][-1],
                         'cost_gap_to_analytical': float(result['cost_history'][-1]-reference_J),
                         'stop_reason': stop_description(result), 'split': 'train'})
        np.savez_compressed(res / f'q4_{run_id}_history.npz',
                            theta_final=result['theta_final'], theta_history=result['theta_history'],
                            cost_history=result['cost_history'], learning_rate=rate,
                            converged=result['converged'], status=result['status'],
                            n_iter=result['n_iter'], stop_reason=stop_description(result))

    baseline = runs['baseline']
    theta = baseline['theta_final']
    hist, costs = baseline['theta_history'], baseline['cost_history']
    gradients = np.array([compute_gradient(X, y, t) for t in hist])
    gradient_norms = np.linalg.norm(gradients, axis=1)
    cost_checks = np.array([compute_cost(X, y, t) for t in hist])
    pred_train = predict(X, theta)
    max_pred_diff = float(np.max(np.abs(pred_train-predict(X, reference))))
    check('baseline_converged', baseline['status'] == 'converged' and baseline['converged'])
    check('history_shape_and_length', hist.shape == (baseline['n_iter']+1, 2) and len(costs) == len(hist))
    check('history_includes_initial_state', np.array_equal(hist[0], THETA_INITIAL) and abs(costs[0]-compute_cost(X, y, THETA_INITIAL)) < 1e-10)
    check('history_ends_at_final_theta', np.array_equal(hist[-1], theta))
    check('all_history_values_finite', np.isfinite(hist).all() and np.isfinite(costs).all())
    check('all_history_costs_match_parameters', np.allclose(cost_checks, costs, atol=1e-10, rtol=0))
    check('all_steps_follow_GD_rule', np.allclose(hist[1:], hist[:-1]-BASELINE_RATE*gradients[:-1], atol=1e-12, rtol=0))
    check('baseline_cost_nonincreasing_with_roundoff_tolerance', np.all(np.diff(costs) <= 1e-9))
    check('final_gradient_meets_stopping_tolerance', gradient_norms[-1] <= TOLERANCE)
    check('previous_gradient_not_yet_converged', baseline['n_iter'] == 0 or gradient_norms[-2] > TOLERANCE)
    check('recorded_gradient_matches_final_state', abs(baseline['gradient_norm']-gradient_norms[-1]) < 1e-12)
    check('GD_predictions_match_independent_reference', max_pred_diff <= PREDICTION_TOLERANCE_MW)
    check('MSE_train_equals_twice_cost', np.isclose(np.mean((y-pred_train)**2), 2*costs[-1], atol=1e-10, rtol=0))

    # Freeze the train-defined baseline before producing any test predictions.
    test = pd.read_csv(ROOT / 'data/processed/test.csv')
    if len(test) != contract['n_test']:
        raise ValueError('Unexpected test sample count.')
    check('train_and_test_source_rows_disjoint', set(train.source_row).isdisjoint(test.source_row))
    Xtest = design_matrix(test.AT.to_numpy(), scaling)
    pred_test = predict(Xtest, theta)
    check('test_prediction_shape_and_values', pred_test.shape == (contract['n_test'],) and np.isfinite(pred_test).all())
    beta = np.array([theta[0]-theta[1]*scaling['mean_AT']/scaling['std_AT'], theta[1]/scaling['std_AT']])
    check('raw_scale_coefficients_preserve_train_predictions', np.allclose(beta[0]+beta[1]*train.AT.to_numpy(), pred_train, atol=1e-10, rtol=0))

    # Required baseline fields plus documented, optional experiment fields allow
    # the TV4 learning-rate reader to receive the same runs on real shared data.
    handoff = {'theta_final': theta, 'theta_history': hist, 'cost_history': costs,
               'pred_train': pred_train, 'pred_test': pred_test,
               'source_row_train': train.source_row.to_numpy(dtype=np.int64),
               'source_row_test': test.source_row.to_numpy(dtype=np.int64),
               'theta_analytical': reference, 'cost_analytical': reference_J}
    for alias, run_id in [('small', 'small'), ('appropriate', 'baseline'), ('large', 'large'), ('divergent', 'divergent')]:
        r = runs[run_id]
        handoff.update({f'lr_{alias}': r['learning_rate'], f'theta_history_{alias}': r['theta_history'],
                        f'cost_history_{alias}': r['cost_history'], f'converged_{alias}': r['converged'],
                        f'stop_reason_{alias}': stop_description(r)})
    np.savez_compressed(res / 'baseline_run.npz', **handoff)
    config = {'run_id': 'baseline', 'data_version': contract['data_version'], 'is_mock_data': False,
              'theta_initial': THETA_INITIAL.tolist(), 'theta_init': THETA_INITIAL.tolist(),
              'learning_rate': BASELINE_RATE, 'max_iter': MAX_ITER, 'tolerance': TOLERANCE,
              'n_iter': baseline['n_iter'], 'actual_iterations': baseline['n_iter'],
              'status': baseline['status'], 'converged': baseline['converged'],
              'stop_rule': 'L2 norm of full-train gradient <= tolerance',
              'stop_reason': stop_description(baseline), 'gradient_norm_final': baseline['gradient_norm'],
              'divergence_rule': 'J > 1e12 * max(J_initial, 1), or nonfinite gradient/candidate',
              'baseline_choice': 'alpha=0.1 fixed for a readable multi-step trajectory, not claimed to be fastest',
              'n_train': len(train), 'n_test': len(test),
              'history_index': 'k means state after k accepted updates; step 0 is initial state',
              'train_sha256': contract['sha256']['data/processed/train.csv'],
              'test_sha256': contract['sha256']['data/processed/test.csv'],
              'scaling_sha256': contract['sha256']['results/scaling.json'],
              'experiment_runs': {r['run_id']: {'learning_rate': r['learning_rate'], 'n_iter': r['n_iter'], 'status': r['status']} for r in run_rows},
              'optional_npz_fields': 'source-row arrays, independent reference and four LR histories; never required for basic prediction',
              'test_predictions_exported': True, 'test_metrics_computed': False}
    write_json(res / 'baseline_config.json', config)
    pd.DataFrame(run_rows).to_csv(res / 'q4_learning_rate_runs.csv', index=False)
    pd.DataFrame({'iteration': np.arange(len(hist)), 'theta0_scaled': hist[:, 0],
                  'theta1_scaled': hist[:, 1], 'J_train': costs,
                  'gradient_l2_norm': gradient_norms}).to_csv(res / 'q4_baseline_trace.csv', index=False)
    summary = {'question': 4, 'data_version': contract['data_version'],
               'status': baseline['status'], 'n_iter': baseline['n_iter'], 'history_states': len(hist),
               'theta_initial': THETA_INITIAL.tolist(), 'theta_final_scaled': theta.tolist(),
               'intercept_raw_MW': beta[0], 'slope_raw_MW_per_degC': beta[1],
               'J_initial': costs[0], 'J_final_train': costs[-1],
               'MSE_final_train': float(np.mean((y-pred_train)**2)),
               'gradient_l2_final': gradient_norms[-1], 'reference_theta_scaled': reference.tolist(),
               'reference_J_train': reference_J, 'max_pred_difference_reference_train_MW': max_pred_diff,
               'prediction_tolerance_MW': PREDICTION_TOLERANCE_MW,
               'max_history_cost_recalculation_error': float(np.max(np.abs(cost_checks-costs))),
               'checks_passed_before_archive_check': len(checks),
               'test_metrics_computed': False,
               'scope': 'Q4 training and handoff. Analytical oracle is an independent verification aid; Q5 derivation and comparison remain separate.'}
    write_json(res / 'q4_summary.json', summary)
    with np.load(res / 'baseline_run.npz', allow_pickle=False) as saved:
        check('saved_baseline_arrays_match', all(np.array_equal(saved[k], handoff[k]) for k in ['theta_final', 'theta_history', 'cost_history', 'pred_train', 'pred_test']))
        check('saved_LR_fields_available_for_TV4', all(f'lr_{tag}' in saved.files and f'cost_history_{tag}' in saved.files for tag in ['small', 'appropriate', 'large', 'divergent']))
    verify_shared_files()
    check('shared_data_files_unchanged', True)
    pd.DataFrame(checks).to_csv(res / 'q4_verification.csv', index=False)
    summary['checks_passed'] = len(checks)
    write_json(res / 'q4_summary.json', summary)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(len(costs)), costs, color='#24678C', lw=2)
    ax.scatter([0, baseline['n_iter']], [costs[0], costs[-1]], color=['#BB4D28', '#28794C'], s=45, zorder=5)
    ax.set(xlabel='Accepted Gradient Descent updates k', ylabel='Training cost J (MW²)',
           title=f'Q4 baseline | learning rate 0.1 | {baseline["n_iter"]} updates')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(fig_dir / 'q4_baseline_cost.png', dpi=180)
    plt.close(fig)
    print(pd.DataFrame(run_rows)[['run_id', 'learning_rate', 'n_iter', 'status', 'J_final']].to_string(index=False))
    print('Baseline theta (standardized coordinates):', theta.tolist())
    print(f'Raw model: PE_hat = {beta[0]:.9f} {beta[1]:+.9f} * AT')
    print(f'J_train: {costs[0]:.9f} -> {costs[-1]:.9f}')
    print(f'Final gradient norm: {gradient_norms[-1]:.12g}')
    print(f'Max prediction difference vs analytical reference on train: {max_pred_diff:.12g} MW')
    print(f'Q4 CHECK: PASS ({len(checks)} checks)')
    print('Shared data unchanged. Test predictions exported; no test metrics computed.')

if __name__ == '__main__':
    main()
