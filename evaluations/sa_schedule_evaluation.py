import numpy as np
from matplotlib import pyplot as plt

from algorithms.simulated_annealing import SimulatedAnnealing
from objective_functions.rastrigin import RastriginObjective


def run_experiment():
    schedules = {
        'linear': {'alpha': 100},
        'log': {},
        'exponential': {'beta': 0.98},
        'adaptive': {
            'window': 50, 'rate_target': 0.4,
            'alpha_fast': 0.90, 'alpha_slow': 0.99
        },
        'custom': {'alpha': 0.01, 'beta': 0.95, 'switch_point': 150}
    }
    runs = 100
    max_iter = 10000
    tol = 1.0
    dim = 8
    init_temp = 100000
    step_size = 0.1
    results = {}

    for schedule, params in schedules.items():
        best_vals, iters_to_tol, accept_rates = [], [], []

        for _ in range(runs):
            obj = RastriginObjective(dim=dim)
            sa = SimulatedAnnealing(
                objective=obj,
                initial_temp=init_temp,
                schedule=schedule,
                schedule_params=params,
                step_size=step_size,
                max_iter=max_iter
            )
            sa.run()
            summary = sa.summary()
            best_vals.append(summary['best_evaluation'])

            for i, val in enumerate(summary['history']):
                if val <= tol:
                    iters_to_tol.append(i)
                    break
            else:
                iters_to_tol.append(max_iter)

            accept_rates.append(sa.accept_history)

        results[schedule] = {
            'best_vals': best_vals,
            'iters_to_tol': iters_to_tol,
            'accept_rates': accept_rates
        }

    return results

def plot_all(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. Boxplot: Solution Quality ---
    axes[0].boxplot([v['best_vals'] for v in results.values()], labels=results.keys())
    axes[0].set_title("Solution Quality")
    axes[0].set_ylabel("Best Final Evaluation")
    axes[0].grid(True)

    # --- 2. Barplot: Convergence Speed ---
    means = [np.mean(v['iters_to_tol']) for v in results.values()]
    axes[1].bar(results.keys(), means)
    axes[1].set_title("Convergence Speed")
    axes[1].set_ylabel("Avg. Iterations to Reach Tolerance")
    axes[1].grid(True)

    # --- 3. Lineplot: Acceptance Rate ---
    for label, val in results.items():
        accept_array = np.array(val['accept_rates'])
        avg_accept = np.mean(accept_array, axis=0)
        smoothed = np.convolve(avg_accept, np.ones(10)/10, mode='valid')
        axes[2].plot(smoothed, label=label)
    axes[2].set_title("Exploration Rate")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Acceptance Rate (Moving Avg)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
