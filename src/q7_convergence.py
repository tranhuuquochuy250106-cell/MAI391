"""Analyze convergence using the verified Q4 train histories and Q5 optimum."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_io import ROOT, load_training_data
from .linear_model import compute_cost, compute_gradient

COLORS = {
    'small': '#7b4592',
    'baseline': '#146380',
    'large': '#ba5b11',
    'divergent': '#b63440',
    'one_step': '#218359',
}


def plot_baseline(costs, gaps, j_star, output):
    k = np.arange(len(costs))
    fig, (cost_ax, gap_ax) = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    color = COLORS['baseline']
    cost_ax.plot(k, costs, color=color, linewidth=2.3, label='Chi phí từng bước')
    cost_ax.axhline(j_star, color='#485565', linestyle='--', label=r'Nghiệm giải tích $J^*$')
    cost_ax.legend(frameon=False)
    cost_ax.set(ylabel=r'$J_{\mathrm{train}}$ (MW²)', title='Chi phí từ bước khởi tạo')
    gap_ax.semilogy(k, np.where(gaps > 0, gaps, np.nan), color=color, linewidth=2.3)
    gap_ax.scatter(k[-1], gaps[-1], color=color, zorder=3)
    gap_ax.annotate(f'Dừng tại bước {k[-1]}', (k[-1], gaps[-1]),
                    xytext=(-8, 14), textcoords='offset points', ha='right')
    gap_ax.set(ylabel=r'Khoảng cách tới tối ưu $\Delta J$ (MW²)',
               title='Khoảng cách tới nghiệm tối ưu · trục log')
    for ax in (cost_ax, gap_ax):
        ax.set_xlabel('Số lần cập nhật')
        ax.grid(alpha=0.22)
    fig.suptitle('Câu 7 · Hội tụ của Gradient Descent cơ sở', fontsize=15, fontweight='bold')
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_learning_rates(traces, output):
    fig, (gap_ax, direction_ax) = plt.subplots(1, 2, figsize=(12, 4.9), layout='constrained')
    for run_id, (rate, gap, signed_error) in traces.items():
        color = COLORS[run_id]
        gap_ax.semilogy(np.arange(len(gap)), np.where(gap > 0, gap, np.nan),
                        color=color, linewidth=1.9, label=f'γ = {rate:g}', marker='.',
                        markevery=[0, len(gap) - 1])
        if run_id != 'one_step':
            first = signed_error[:13]
            direction_ax.plot(np.arange(len(first)), first, color=color,
                              linewidth=1.65, marker='o', markersize=3.5)
    direction_ax.axhline(0, color='#485565', linestyle='--', linewidth=1)
    gap_ax.legend(frameon=False, ncol=2)
    gap_ax.set(title='Khoảng cách tới tối ưu theo learning rate',
               ylabel=r'$\Delta J$ (MW²), trục log')
    direction_ax.set(title='Vị trí hai phía của nghiệm · 12 cập nhật đầu',
                     ylabel=r'$(\theta_0^{(k)}-\theta_0^*) / (\theta_0^{(0)}-\theta_0^*)$',
                     xticks=np.arange(0, 13, 2))
    for ax in (gap_ax, direction_ax):
        ax.set_xlabel('Số lần cập nhật')
        ax.grid(alpha=0.22)
    fig.suptitle('Câu 7 · Ảnh hưởng của learning rate', fontsize=15, fontweight='bold')
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    X, y, train, _ = load_training_data()
    results = ROOT / 'results'
    config = json.loads((results / 'baseline_config.json').read_text(encoding='utf-8'))
    q5 = json.loads((results / 'q5_summary.json').read_text(encoding='utf-8'))
    contract = json.loads((results / 'data_contract.json').read_text(encoding='utf-8'))
    if (config['is_mock_data'] or not config['converged']
            or config['data_version'] != q5['data_version']
            or config['data_version'] != contract['data_version']
            or config['train_sha256'] != contract['sha256']['data/processed/train.csv']
            or config['scaling_sha256'] != contract['sha256']['results/scaling.json']):
        raise ValueError('Expected Q4 and Q5 results on the verified train split.')

    optimum = np.array(q5['theta_scaled'])
    j_star = compute_cost(X, y, optimum)
    hessian = X.T @ X / len(y)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.linalg.norm(compute_gradient(X, y, optimum)) > 1e-9:
        raise ValueError('Q5 reference fails the stationary-point check.')

    profiles = pd.read_csv(results / 'q4_learning_rate_runs.csv').set_index('run_id')
    traces = {}
    rows = []
    for run_id, spec in config['experiment_runs'].items():
        rate, steps, status = spec['learning_rate'], spec['n_iter'], spec['status']
        profile = profiles.loc[run_id]
        np.testing.assert_allclose(
            [profile.learning_rate, profile.n_iter, profile.max_iter,
             profile.tolerance_gradient_l2, profile.theta0_initial, profile.theta1_initial],
            [rate, steps, config['max_iter'], config['tolerance'], *config['theta_initial']],
        )
        if profile.status != status or profile.split != 'train':
            raise ValueError(f'{run_id}: inconsistent Q4 metadata.')
        with np.load(results / f'q4_{run_id}_history.npz', allow_pickle=False) as run:
            history, costs = run['theta_history'], run['cost_history']
            np.testing.assert_array_equal(history[-1], run['theta_final'])
            if (run['learning_rate'].item(), run['n_iter'].item(), run['status'].item()) != (
                rate, steps, status
            ):
                raise ValueError(f'{run_id}: history metadata mismatch.')
        if history.shape != (steps + 1, 2) or costs.shape != (steps + 1,):
            raise ValueError(f'{run_id}: inconsistent history length.')
        np.testing.assert_array_equal(history[0], config['theta_initial'])
        gradients = np.array([compute_gradient(X, y, theta) for theta in history])
        direct_costs = np.array([compute_cost(X, y, theta) for theta in history])
        np.testing.assert_allclose(costs, direct_costs, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(history[1:], history[:-1] - rate * gradients[:-1],
                                   atol=1e-10, rtol=1e-12)
        norms = np.linalg.norm(gradients, axis=1)
        if status == 'converged':
            if norms[-1] > config['tolerance'] or np.any(norms[:-1] <= config['tolerance']):
                raise ValueError(f'{run_id}: inconsistent gradient stopping criterion.')
        else:
            threshold = 1e12 * max(costs[0], 1)
            if status != 'diverged' or costs[-1] <= threshold or np.any(costs[:-1] > threshold):
                raise ValueError(f'{run_id}: inconsistent divergence stopping criterion.')

        delta = history - optimum
        # Quadratic optimality gap avoids cancellation in J(theta) - J*.
        gaps = 0.5 * np.einsum('ki,ij,kj->k', delta, hessian, delta)
        np.testing.assert_allclose(costs, j_star + gaps, atol=1e-10, rtol=1e-12)
        signed_error = delta[:, 0] / delta[0, 0]
        traces[run_id] = (rate, gaps, signed_error)
        rows.append({
            'run_id': run_id, 'learning_rate': rate, 'n_train': len(train),
            'theta0_initial': history[0, 0], 'theta1_initial': history[0, 1],
            'max_iter': config['max_iter'], 'gradient_tolerance': config['tolerance'],
            'n_iter': steps, 'status': status, 'stop_reason': profile.stop_reason,
            'gradient_norm_final': norms[-1], 'J_initial': costs[0], 'J_final': costs[-1],
            'J_analytical': j_star, 'optimality_gap_final_MW2': gaps[-1],
            'theta0_sign_flips_first_12_updates': int(np.count_nonzero(
                np.signbit(signed_error[1:13]) != np.signbit(signed_error[:12])
            )) if run_id != 'one_step' else None,
        })
        if run_id == 'baseline':
            with np.load(results / 'baseline_run.npz', allow_pickle=False) as baseline:
                np.testing.assert_array_equal(baseline['source_row_train'], train.source_row)
                np.testing.assert_array_equal(history, baseline['theta_history'])
                np.testing.assert_array_equal(costs, baseline['cost_history'])
            plot_baseline(costs, gaps, j_star, ROOT / 'figures/q7_convergence.png')

    plot_learning_rates(traces, ROOT / 'figures/q7_learning_rate.png')
    runs = pd.DataFrame(rows)
    runs['theta0_sign_flips_first_12_updates'] = runs[
        'theta0_sign_flips_first_12_updates'
    ].astype('Int64')
    runs.to_csv(results / 'q7_runs.csv', index=False)
    summary = {
        'data_version': config['data_version'], 'n_train': len(train), 'split': 'train',
        'hessian_eigenvalues': eigenvalues.tolist(),
        'strict_stability_upper_bound': float(2 / eigenvalues[-1]),
        'j_analytical': j_star, 'runs_verified': len(rows),
        'states_verified': sum(row['n_iter'] + 1 for row in rows),
        'verification': 'PASS',
    }
    (results / 'q7_summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                            encoding='utf-8')
    print(f"Q7 CHECK: PASS ({len(rows)} runs, {summary['states_verified']} states verified)")


if __name__ == '__main__':
    main()
