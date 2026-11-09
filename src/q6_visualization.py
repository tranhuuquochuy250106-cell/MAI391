"""Plot the verified Q4 trajectory on the Q3 training cost landscape."""
import json

import matplotlib.pyplot as plt
import numpy as np

from .data_io import ROOT, load_training_data
from .linear_model import compute_cost, compute_gradient

PATH = '#ad301b'
START = '#f6ac43'
END = '#123957'


def cost_landscape(parameters, optimum, j_star, gradient, hessian):
    """Evaluate the exact quadratic expansion of half-MSE around the reference."""
    delta = parameters - optimum
    return j_star + delta @ gradient + 0.5 * np.einsum(
        '...i,ij,...j->...', delta, hessian, delta
    )


def make_grid(xlim, ylim, evaluate):
    x, y = np.meshgrid(np.linspace(*xlim, 161), np.linspace(*ylim, 161))
    return x, y, evaluate(np.stack((x, y), axis=-1))


def plot_surface(grid, history, cost, output):
    fig = plt.figure(figsize=(10.6, 7.3), layout='constrained')
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    surface = ax.plot_surface(*grid, cmap='Blues', alpha=0.53,
                              linewidth=0, rcount=85, ccount=85, zorder=1)
    ax.plot(*history.T, cost, color=PATH, linewidth=2.5, zorder=3,
            label=f'Quỹ đạo GD · {len(cost)-1} cập nhật')
    for index, color, marker, label in (
        (0, START, '^', 'Khởi tạo'), (-1, END, '*', 'Điểm cuối GD')
    ):
        ax.scatter(*history[index], cost[index], color=color, marker=marker,
                   edgecolors='white', linewidths=0.8, s=155, zorder=5, label=label)
    ax.set(xlabel=r'$\theta_0$ (MW)', ylabel=r'$\theta_1$ (MW trên một đơn vị $z$)',
           zlabel=r'$J_{\mathrm{train}}$ (MW²)')
    ax.view_init(elev=27, azim=-124)
    ax.set_box_aspect((1.25, 0.95, 0.82))
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.09, label=r'$J_{\mathrm{train}}$ (MW²)')
    ax.legend(loc='upper left', frameon=False)
    fig.suptitle('Câu 6 · Bề mặt chi phí và quỹ đạo Gradient Descent',
                 fontsize=15, fontweight='bold')
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_contours(full_grid, zoom_grid, history, optimum, j_star, output):
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(12, 6.1), layout='constrained')
    panels = (
        (full, full_grid, [100, 500, 2000, 8000, 20000, 45000, 80000, 120000, 180000],
         'Toàn bộ quỹ đạo'),
        (zoom, zoom_grid, j_star + np.array([0.5, 2, 8, 18, 32, 50, 72, 98, 130]),
         'Phóng to quanh nghiệm giải tích'),
    )
    for ax, (x, y, cost), levels, title in panels:
        ax.contourf(x, y, cost, levels=np.r_[j_star, levels], cmap='Blues', extend='max')
        contours = ax.contour(x, y, cost, levels=levels, colors='#527894', linewidths=0.75)
        ax.clabel(contours, fmt='%.0f' if ax is full else '%.1f', fontsize=7)
        ax.plot(*history.T, color=PATH, linewidth=2.3, zorder=3)
        ax.scatter(*history[-1], color=END, marker='*', s=145, edgecolors='white',
                   linewidths=0.8, zorder=5)
        ax.set(xlabel=r'$\theta_0$ (MW)', ylabel=r'$\theta_1$ (MW trên một đơn vị $z$)',
               xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), title=title)
        ax.set_aspect('equal', adjustable='box')
        ax.ticklabel_format(useOffset=False, axis='both')
    full.scatter(*history[0], color=START, marker='^', s=115, edgecolors='white', zorder=5)
    full.annotate('Khởi tạo', history[0], xytext=(10, 10), textcoords='offset points')
    full.annotate('Điểm cuối GD', history[-1], xytext=(-7, -18),
                  textcoords='offset points', ha='right')
    zoom.annotate('Điểm cuối GD ≈ nghiệm giải tích', optimum, xytext=(17, 25),
                  textcoords='offset points', fontsize=9,
                  arrowprops={'arrowstyle': '-', 'color': END})
    fig.suptitle('Câu 6 · Đường đồng mức của chi phí train (MW²)',
                 fontsize=15, fontweight='bold')
    fig.savefig(output, dpi=220)
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
        raise ValueError('Expected Q4 and Q5 results on the verified training data.')
    with np.load(results / 'baseline_run.npz', allow_pickle=False) as run:
        history, costs = run['theta_history'], run['cost_history']
        np.testing.assert_array_equal(run['source_row_train'], train.source_row)
        np.testing.assert_array_equal(history[0], config['theta_initial'])
        np.testing.assert_array_equal(history[-1], run['theta_final'])
    states = config['n_iter'] + 1
    if history.shape != (states, 2) or costs.shape != (states,):
        raise ValueError('Expected one parameter pair and cost per saved state.')
    direct_costs = np.array([compute_cost(X, y, theta) for theta in history])
    np.testing.assert_allclose(costs, direct_costs, rtol=1e-12, atol=1e-10)

    optimum = np.asarray(q5['theta_scaled'])
    j_star = compute_cost(X, y, optimum)
    gradient = compute_gradient(X, y, optimum)
    if np.linalg.norm(gradient) > 1e-9:
        raise ValueError('Q5 reference is not stationary on the verified training data.')
    hessian = X.T @ X / len(y)

    def evaluate(parameters):
        return cost_landscape(parameters, optimum, j_star, gradient, hessian)

    np.testing.assert_allclose(evaluate(history), direct_costs, rtol=1e-12, atol=1e-10)
    span = np.ptp(history[:, 0])
    xlim = (history[:, 0].min() - 0.11 * span, history[:, 0].max() + 0.11 * span)
    ylim = (optimum[1] - 0.43 * span, optimum[1] + 0.43 * span)
    full_grid = make_grid(xlim, ylim, evaluate)
    zoom_grid = make_grid((optimum[0] - 13, optimum[0] + 13),
                          (optimum[1] - 13, optimum[1] + 13), evaluate)
    for x, y_grid, grid_cost in (full_grid, zoom_grid):
        for i, j in ((0, 0), (80, 80), (-1, -1)):
            point = np.array([x[i, j], y_grid[i, j]])
            np.testing.assert_allclose(grid_cost[i, j], compute_cost(X, y, point),
                                       rtol=1e-12, atol=1e-10)

    plot_surface(full_grid, history, costs, ROOT / 'figures/q6_surface.png')
    plot_contours(full_grid, zoom_grid, history, optimum, j_star,
                  ROOT / 'figures/q6_contour.png')
    summary = {
        'data_version': config['data_version'],
        'n_train': len(train),
        'n_iter': config['n_iter'],
        'plotted_states': states,
        'theta_start': history[0].tolist(),
        'theta_final': history[-1].tolist(),
        'theta_analytical': optimum.tolist(),
        'j_start': float(costs[0]),
        'j_final': float(costs[-1]),
        'j_analytical': j_star,
        'max_abs_saved_vs_recomputed_cost': float(np.max(np.abs(costs - direct_costs))),
        'max_abs_surface_vs_recomputed_cost_on_path': float(
            np.max(np.abs(evaluate(history) - direct_costs))
        ),
        'verification': 'PASS',
    }
    (results / 'q6_summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                            encoding='utf-8')
    print(f'Q6 CHECK: PASS ({states} states verified)')


if __name__ == '__main__':
    main()
