import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from algorithms.ant_colony_optimization import AntColonyOptimization


def create_distance_matrix(n_cities, seed=None):
    """
    Generate a random symmetric distance matrix for n_cities in 2D.
    """
    rng = np.random.RandomState(seed)
    coords = rng.rand(n_cities, 2) * 100
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    return dist_matrix, coords


def _run_combo(args):
    """
    Worker: run `runs` independent trials for one (alpha, beta, rho, n_ants) combo.
    Returns a list of dicts (one dict per trial).
    """
    distance_matrix, alpha, beta, rho, n_ants, runs, n_iterations = args
    out = []
    for _ in range(runs):
        aco = AntColonyOptimization(
            distance_matrix=distance_matrix,
            n_ants=n_ants,
            n_iterations=n_iterations,
            decay=rho,
            alpha=alpha,
            beta=beta
        )
        _, best_dist = aco.run()
        out.append({
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'n_ants': n_ants,
            'best_dist': best_dist
        })
    return out


def evaluate_aco_hyperparameters(
    distance_matrix,
    alphas,
    betas,
    rhos,
    n_ants_list,
    runs=30,
    n_iterations=100,
    n_jobs=None
):
    """
    Parallel evaluation: one task per hyperparameter combo (each does `runs` trials).
    Returns a DataFrame with one row per trial.
    """
    # 1) build one tuple-per-combo
    combos = [
        (distance_matrix, a, b, rho, ants, runs, n_iterations)
        for a in alphas
        for b in betas
        for rho in rhos
        for ants in n_ants_list
    ]

    records = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_run_combo, combo) for combo in combos]
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="ACO combos"):
            batch = fut.result()
            # EXTEND, not append!
            records.extend(batch)

    df = pd.DataFrame(records)

    # --- optional sanity check ---
    # counts = df.groupby(['alpha','beta','rho','n_ants']).size()
    # assert counts.min() == runs and counts.max() == runs, \
    #        f"Expected {runs} trials per combo but got {counts.unique()}"
    # --------------------------------

    return df


def plot_hyperparameter_effects(df):
    """
    Plot mean ± std errorbars for each hyperparameter.
    """
    params = ['alpha', 'beta', 'rho', 'n_ants']
    fig, axes = plt.subplots(1, len(params), figsize=(20, 5), sharey=True)

    for ax, p in zip(axes, params):
        grp = df.groupby(p)['best_dist']
        x = grp.mean().index.astype(float)
        y = grp.mean().values
        yerr = grp.std().values

        ax.errorbar(x, y, yerr=yerr, marker='o', linestyle='-')
        ax.set_title(f'Effect of {p}')
        ax.set_xlabel(p)
        ax.set_ylabel('Best Distance')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def print_parameter_rankings(df, top_k=5):
    """
    Print the top and bottom k hyperparameter settings by mean best_dist.
    """
    stats = (
        df.groupby(['alpha','beta','rho','n_ants'])['best_dist']
          .agg(mean='mean', std='std')
          .reset_index()
          .sort_values('mean')
    )

    print(f"\nTop {top_k} parameter combos:")
    for _, row in stats.head(top_k).iterrows():
        a, b, r, ants = row['alpha'], row['beta'], row['rho'], int(row['n_ants'])
        m, s = row['mean'], row['std']
        print(f" α={a}, β={b}, ρ={r}, ants={ants} -> mean={m:.2f}, std={s:.2f}")

    print(f"\nWorst {top_k} parameter combos:")
    for _, row in stats.tail(top_k).iterrows():
        a, b, r, ants = row['alpha'], row['beta'], row['rho'], int(row['n_ants'])
        m, s = row['mean'], row['std']
        print(f" α={a}, β={b}, ρ={r}, ants={ants} -> mean={m:.2f}, std={s:.2f}")


def analyze_convergence_behavior(df, top_k=5):
    """
    Identify the most unstable combos by coefficient of variation (std/mean).
    """
    cv = (
        df.groupby(['alpha','beta','rho','n_ants'])['best_dist']
          .agg(mean='mean', std='std')
          .reset_index()
    )
    cv['cv'] = cv['std'] / cv['mean']
    unstable = cv.sort_values('cv', ascending=False).head(top_k)

    print(f"\nTop {top_k} most unstable combos (by CV):")
    for _, row in unstable.iterrows():
        a, b, r, ants = row['alpha'], row['beta'], row['rho'], int(row['n_ants'])
        m, s, c = row['mean'], row['std'], row['cv']
        print(f" α={a}, β={b}, ρ={r}, ants={ants} -> mean={m:.2f}, std={s:.2f}, cv={c:.3f}")


def run_aco_experiment():
    # 1) problem setup
    dist_matrix, _ = create_distance_matrix(n_cities=20, seed=42)

    # 2) hyperparameter grid
    alphas      = [0.5, 1.0, 1.5, 2.0]
    betas       = [1.0, 2.0, 3.0, 4.0]
    rhos        = [0.1, 0.3, 0.5, 0.7]
    n_ants_list = [10, 20, 30, 50]

    # 3) evaluate
    df = evaluate_aco_hyperparameters(
        dist_matrix, alphas, betas, rhos, n_ants_list,
        runs=30, n_iterations=100, n_jobs=None
    )

    # 4) print & analyze
    print(df.groupby(['alpha','beta','rho','n_ants'])['best_dist'].agg(['size','mean','std']))
    print_parameter_rankings(df)
    analyze_convergence_behavior(df)

    # 5) plot
    plot_hyperparameter_effects(df)