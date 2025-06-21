import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.tabu_search import TabuSearch
from objective_functions.rastrigin import RastriginObjective



def evaluate_tabu_hyperparams():
    # Experiment setup
    dim = 2
    ras = RastriginObjective(dim)
    bounds = [ras.bounds] * dim
    threshold = 1.0      # convergence threshold on f(x)
    n_runs = 40          # independent trials per configuration
    max_iters = 10000     # max iterations per run

    # Hyperparameter grid
    tabu_tenures = [5, 10, 20]
    step_sizes   = [0.05, 0.1, 0.2]
    neigh_sizes  = [10, 20]


    records = []
    # Sweep hyperparameters
    for T in tabu_tenures:
        for s in step_sizes:
            for k in neigh_sizes:
                final_vals = []
                conv_iters = []
                for run in range(n_runs):
                    ts = TabuSearch(func=ras.evaluate,
                                    bounds=bounds,
                                    max_iters=max_iters,
                                    tabu_tenure=T,
                                    neighborhood_size=k,
                                    step_size=s)
                    _, val, hist = ts.run()
                    final_vals.append(val)
                    # find iteration of first convergence
                    conv_it = next((i for i, v in enumerate(hist) if v < threshold), np.nan)
                    conv_iters.append(conv_it)

                finite_iters = [it for it in conv_iters if np.isfinite(it)]
                records.append({
                    'tabu_tenure': T,
                    'step_size'  : s,
                    'neigh_size' : k,
                    'avg_final'  : np.mean(final_vals),
                    'std_final'  : np.std(final_vals),
                    'conv_rate'  : len(finite_iters) / n_runs,
                    'avg_conv_it': np.mean(finite_iters) if finite_iters else np.nan
                })

    df_ts = pd.DataFrame(records)
    print("Tabu Search Hyperparameter Results:")
    print(df_ts)
    df_ts.to_csv('tabu_search_hyperparam_results.csv', index=False)

    # For neigh_size = 20
    subset = df_ts[df_ts['neigh_size'] == 20]

    # Pivot for avg_final and conv_rate
    pivot_avg = subset.pivot(index='tabu_tenure', columns='step_size', values='avg_final')
    pivot_rate = subset.pivot(index='tabu_tenure', columns='step_size', values='conv_rate')

    # Common extent and ticks
    x0, x1 = -0.5, len(step_sizes) - 0.5
    y0, y1 = -0.5, len(tabu_tenures) - 0.5
    extent = [x0, x1, y0, y1]
    xticks = range(len(step_sizes))
    yticks = range(len(tabu_tenures))

    # Plot avg_final heatmap
    plt.figure(figsize=(5,4))
    plt.title("Tabu Search Avg Final Fitness (neigh_size=20)")
    im = plt.imshow(pivot_avg.values, origin='lower', extent=extent, aspect='auto')
    plt.xticks(xticks, step_sizes)
    plt.yticks(yticks, tabu_tenures)
    plt.xlabel('step_size')
    plt.ylabel('tabu_tenure')
    plt.colorbar(im, label='Avg Final f')
    plt.tight_layout()
    plt.show()

    # Plot conv_rate heatmap
    plt.figure(figsize=(5,4))
    plt.title("Tabu Search Convergence Rate (neigh_size=20)")
    im2 = plt.imshow(pivot_rate.values, origin='lower', extent=extent, vmin=0, vmax=1, aspect='auto')
    plt.xticks(xticks, step_sizes)
    plt.yticks(yticks, tabu_tenures)
    plt.xlabel('step_size')
    plt.ylabel('tabu_tenure')
    plt.colorbar(im2, label='Convergence Rate')
    plt.tight_layout()
    plt.show()
