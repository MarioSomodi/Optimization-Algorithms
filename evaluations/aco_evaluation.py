import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from algorithms.ant_colony_optimization import AntColonyOptimization

class AcoEvaluator:
    """
        distance_matrix – symmetric matrix of pairwise distances between cities
        alphas – list of pheromone influence exponents to test (α)
        betas – list of heuristic influence exponents to test (β)
        rhos - list of evaporation rates to test (ρ)
        n_ants_list - list of colony sizes (number of ants) to test
        runs - number of independent ACO trials per combination
        n_iterations - number of iterations (colony cycles) per trial
        df - results DataFrame after evaluate(), one row per trial
    """
    def __init__(
        self,
        distance_matrix,
        alphas,
        betas,
        rhos,
        n_ants_list,
        runs=30,
        n_iterations=100,
    ):
        self.distance_matrix = distance_matrix
        self.alphas = alphas
        self.betas = betas
        self.rhos = rhos
        self.n_ants_list = n_ants_list
        self.runs = runs
        self.n_iterations = n_iterations
        self.df = None

    @staticmethod
    def create_distance_matrix(n_cities, seed=None):
        """
        Generate a random symmetric distance matrix for n_cities in 2D.
        Returns (distance_matrix, coords).
        """
        rng = np.random.RandomState(seed)
        coords = rng.rand(n_cities, 2) * 100
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
        return dist_matrix, coords

    @staticmethod
    def _run_combo(args):
        """
        unpack args, run one ACO experiment, return metrics.
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
            aco.run()
            out.append({
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'n_ants': n_ants,
                'best_dist': aco.best_distance
            })
        return out

    def evaluate(self):
        """
        evaluation over all hyperparameter combinations.
        """
        combos = [
            (self.distance_matrix, a, b, rho, ants, self.runs, self.n_iterations)
            for a in self.alphas
            for b in self.betas
            for rho in self.rhos
            for ants in self.n_ants_list
        ]

        records = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._run_combo, combo) for combo in combos]
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Evaluating"):
                records.extend(fut.result())

        self.df = pd.DataFrame(records)

    def plot_hyperparameter_effects(self):
        """
        Plot mean ± std error bars of best distance vs each hyperparameter.
        """
        if self.df is None:
            raise RuntimeError("Call evaluate() before plotting.")

        params = ['alpha', 'beta', 'rho', 'n_ants']
        fig, axes = plt.subplots(1, len(params), figsize=(20, 5), sharey=True)

        for ax, p in zip(axes, params):
            grp = self.df.groupby(p)['best_dist']
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

    def plot_best_and_worst_params(self, top_k=5):
        """
        Bar charts of the top_k and worst_k parameter combos by mean best_dist.
        """
        if self.df is None:
            raise RuntimeError("Call evaluate() before plotting.")

        stats = (
            self.df
              .groupby(['alpha','beta','rho','n_ants'])['best_dist']
              .agg(mean='mean', std='std')
              .reset_index()
              .sort_values('mean')
        )

        top = stats.head(top_k)
        worst = stats.tail(top_k)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Top combos
        labels_top = [
            f"α={a:.1f},β={b:.1f},ρ={r:.1f},ants={int(ants)}"
            for a,b,r,ants in zip(top.alpha, top.beta, top.rho, top.n_ants)
        ]
        x_top = np.arange(len(labels_top))
        ax1.bar(x_top, top['mean'], yerr=top['std'], capsize=5)
        ax1.set_title(f"Top {top_k} Combos (Lowest Mean)")
        ax1.set_xticks(x_top)
        ax1.set_xticklabels(labels_top, rotation=45, ha='right')

        # Worst combos
        labels_worst = [
            f"α={a:.1f},β={b:.1f},ρ={r:.1f},ants={int(ants)}"
            for a,b,r,ants in zip(worst.alpha, worst.beta, worst.rho, worst.n_ants)
        ]
        x_worst = np.arange(len(labels_worst))
        ax2.bar(x_worst, worst['mean'], yerr=worst['std'], capsize=5, color='salmon')
        ax2.set_title(f"Worst {top_k} Combos (Highest Mean)")
        ax2.set_xticks(x_worst)
        ax2.set_xticklabels(labels_worst, rotation=45, ha='right')

        ax1.set_ylabel("Mean Best Distance")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def plot_unstable(self, top_k=5):
        """
        Bar chart of the most unstable combos by coefficient of variation (std/mean).
        """
        if self.df is None:
            raise RuntimeError("Call evaluate() before plotting.")

        cv = (
            self.df
            .groupby(['alpha', 'beta', 'rho', 'n_ants'])['best_dist']
            .agg(mean='mean', std='std')
            .reset_index()
        )
        cv['cv'] = cv['std'] / cv['mean']
        unstable = cv.sort_values('cv', ascending=False).head(top_k)

        labels = [
            f"α={a:.1f},β={b:.1f},ρ={r:.1f},ants={int(ants)}"
            for a, b, r, ants in zip(unstable.alpha, unstable.beta, unstable.rho, unstable.n_ants)
        ]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, unstable['cv'], color='orange')
        plt.title(f"Top {top_k} Most Unstable Combos (CV)")
        plt.ylabel("Coefficient of Variation")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_route(self, coords, route):
        """
        Plot city coordinates and draw the given tour.

        coords : array of shape (n_cities, 2) with x,y positions
        route  : list of city indices in visitation order
        """
        x, y = coords[:, 0], coords[:, 1]
        plt.figure(figsize=(6, 6))
        # cities
        plt.scatter(x, y, s=50, c='black')
        for idx, (xi, yi) in enumerate(coords):
            plt.text(xi, yi, str(idx), color='blue', fontsize=9, ha='right', va='bottom')

        # tour path
        ordered = route + [route[0]]
        rx = coords[ordered, 0]
        ry = coords[ordered, 1]
        plt.plot(rx, ry, '-o', c='red')

        plt.title("ACO Tour")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()