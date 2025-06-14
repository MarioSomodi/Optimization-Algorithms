import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.ant_colony_optimization import AntColonyOptimization

def create_distance_matrix(n_cities, seed=None):
    """
    Generate a random symmetric distance matrix for n_cities in 2D.

    Returns:
        distance_matrix (np.ndarray): Pairwise Euclidean distances.
        coords (np.ndarray): Coordinates of each city.
    """
    rng = np.random.RandomState(seed)
    coords = rng.rand(n_cities, 2) * 100
    # pairwise Euclidean distances
    dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return dist_matrix, coords


def evaluate_aco_hyperparameters(
    distance_matrix,
    alphas,
    betas,
    rhos,
    n_ants_list,
    runs=30,
    n_iterations=100
):
    """
    Run ACO for each combination of hyperparameters multiple times.

    Returns:
        pd.DataFrame: columns [alpha, beta, rho, n_ants, avg_best_dist, std_best_dist]
    """
    records = []
    for alpha in alphas:
        for beta in betas:
            for rho in rhos:
                for n_ants in n_ants_list:
                    best_dists = []
                    # multiple trials for statistics
                    for _ in range(runs):
                        aco = AntColonyOptimization(
                            distance_matrix,
                            n_ants=n_ants,
                            n_iterations=n_iterations,
                            decay=rho,
                            alpha=alpha,
                            beta=beta
                        )
                        _, best_dist = aco.run()
                        best_dists.append(best_dist)
                    records.append({
                        'alpha': alpha,
                        'beta': beta,
                        'rho': rho,
                        'n_ants': n_ants,
                        'avg_best_dist': np.mean(best_dists),
                        'std_best_dist': np.std(best_dists)
                    })
    return pd.DataFrame.from_records(records)


def plot_hyperparameter_effects(df):
    """
    Plot the aggregated effect of each hyperparameter on the average best distance.
    """
    params = ['alpha', 'beta', 'rho', 'n_ants']
    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5), sharey=True)
    for ax, param in zip(axes, params):
        stats = df.groupby(param)['avg_best_dist'].agg(['mean', 'std']).reset_index()
        ax.errorbar(
            stats[param],
            stats['mean'],
            yerr=stats['std'],
            marker='o',
            linestyle='-'
        )
        ax.set_title(f'Effect of {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Avg Best Distance')
        ax.grid(True)
    plt.tight_layout()
    plt.show()