import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from objective_functions.rastrigin import RastriginObjective


def evaluate_pso_hyperparameters(c1_values, c2_values, w_values, runs=30, max_iter=100, tol=1e-3, dim=2):
    """
    Run PSO for each combination of c1, c2, and w multiple times,
    collecting average metrics and convergence histories.
    """
    results = {}

    for w in w_values:
        for c1 in c1_values:
            for c2 in c2_values:
                key = f"w{w}_c1{c1}_c2{c2}"
                best_scores = []
                iters_to_tol = []
                times = []
                histories = []

                for _ in range(runs):
                    obj = RastriginObjective(dim)
                    pso = ParticleSwarmOptimization(
                        objective=obj,
                        n_particles=30,
                        w=w,
                        c1=c1,
                        c2=c2,
                        max_iter=max_iter
                    )
                    _, _ = pso.run()
                    summary = pso.summary()

                    best_scores.append(summary['best_evaluation'])
                    times.append(summary['total_time'])
                    hist = np.array(summary['history'])
                    histories.append(hist)

                    # iteration when solution first <= tol
                    reached = np.where(hist <= tol)[0]
                    iters_to_tol.append(int(reached[0]) if reached.size > 0 else max_iter)

                # aggregate
                avg_best = np.mean(best_scores)
                avg_time = np.mean(times)
                avg_iter = np.mean(iters_to_tol)
                avg_history = np.mean(np.vstack(histories), axis=0)

                results[key] = {
                    'w': w,
                    'c1': c1,
                    'c2': c2,
                    'avg_best': avg_best,
                    'avg_time': avg_time,
                    'avg_iter': avg_iter,
                    'avg_history': avg_history
                }
    return results


def plot_results(results):
    """
    Create a subplot for each weight
    """
    # organize by w
    df = pd.DataFrame([{
        'key': k,
        **v
    } for k, v in results.items()])
    w_values = sorted(df['w'].unique())
    n_w = len(w_values)

    fig, axes = plt.subplots(1, n_w, figsize=(6 * n_w, 5), sharex=True, sharey=True)
    if n_w == 1:
        axes = [axes]

    for ax, w in zip(axes, w_values):
        subset = df[df['w'] == w]
        for _, row in subset.iterrows():
            label = f"c1={row['c1']}, c2={row['c2']}"
            ax.plot(row['avg_history'], alpha=0.7, label=label)
        ax.set_title(f"PSO Convergence (w={w})", fontsize=14)
        ax.set_ylabel('Best-so-far Eval', fontsize=12)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize='small', loc='upper right')

    plt.tight_layout()
    plt.show()
