import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.nelder_mead import NelderMead
from objective_functions.rosenbrock import RosenbrockObjective


def evaluate_nm_hyperparams():
    # Experiment setup
    dim = 2
    ros = RosenbrockObjective(dim)
    threshold = 1e-6  # convergence threshold on f(x)
    n_runs    = 20    # independent trials
    max_iters = 500   # max iterations per run

    # Hyperparameter grid
    alphas = [0.8, 1.0, 1.2]
    gammas = [1.5, 2.0]
    rhos   = [0.3, 0.5]
    sigmas = [0.3, 0.5]

    records_nm = []
    # Sweep hyperparameters
    for alpha in alphas:
        for gamma in gammas:
            for rho in rhos:
                for sigma in sigmas:
                    final_vals = []
                    conv_iters = []
                    for run in range(n_runs):
                        nm = NelderMead(func=ros.evaluate,
                                      x0=np.zeros(dim),
                                      max_iters=max_iters,
                                      tol=threshold,
                                      alpha=alpha,
                                      gamma=gamma,
                                      rho=rho,
                                      sigma=sigma)
                        _, val, hist = nm.run()
                        final_vals.append(val)
                        conv_it = next((i for i, v in enumerate(hist) if v < threshold), np.nan)
                        conv_iters.append(conv_it)

                    finite_iters = [it for it in conv_iters if np.isfinite(it)]
                    records_nm.append({
                        'alpha'      : alpha,
                        'gamma'      : gamma,
                        'rho'        : rho,
                        'sigma'      : sigma,
                        'avg_final'  : np.mean(final_vals),
                        'std_final'  : np.std(final_vals),
                        'conv_rate'  : len(finite_iters) / n_runs,
                        'avg_conv_it': np.mean(finite_iters) if finite_iters else np.nan
                    })

    df_nm = pd.DataFrame(records_nm)
    print("Nelder–Mead Hyperparameter Results:")
    print(df_nm)
    df_nm.to_csv('nelder_mead_hyperparam_results.csv', index=False)

    # Fixed parameters for heatmaps (rho=0.5, sigma=0.5)
    fixed = df_nm[(df_nm['rho'] == 0.5) & (df_nm['sigma'] == 0.5)]
    pivot_avg = fixed.pivot(index='alpha', columns='gamma', values='avg_final')
    pivot_rate = fixed.pivot(index='alpha', columns='gamma', values='conv_rate')

    # Common extent and ticks for NM heatmaps
    x0_nm, x1_nm = -0.5, len(gammas) - 0.5
    y0_nm, y1_nm = -0.5, len(alphas) - 0.5
    extent_nm = [x0_nm, x1_nm, y0_nm, y1_nm]
    xticks_nm = range(len(gammas))
    yticks_nm = range(len(alphas))

    # Avg final fitness heatmap
    plt.figure(figsize=(5,4))
    plt.title("Nelder–Mead Avg Final (rho=0.5, sigma=0.5)")
    im3 = plt.imshow(pivot_avg.values, origin='lower', extent=extent_nm, aspect='auto')
    plt.xticks(xticks_nm, gammas)
    plt.yticks(yticks_nm, alphas)
    plt.xlabel('gamma')
    plt.ylabel('alpha')
    plt.colorbar(im3, label='Avg Final f')
    plt.tight_layout()
    plt.show()

    # Convergence rate heatmap
    plt.figure(figsize=(5,4))
    plt.title("Nelder–Mead Conv Rate (rho=0.5, sigma=0.5)")
    im4 = plt.imshow(pivot_rate.values, origin='lower', extent=extent_nm, vmin=0, vmax=1, aspect='auto')
    plt.xticks(xticks_nm, gammas)
    plt.yticks(yticks_nm, alphas)
    plt.xlabel('gamma')
    plt.ylabel('alpha')
    plt.colorbar(im4, label='Convergence Rate')
    plt.tight_layout()
    plt.show()
