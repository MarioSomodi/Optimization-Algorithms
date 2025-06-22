from itertools import product
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from algorithms.genetic_optimization import GeneticAlgorithm
from objective_functions.rastrigin import RastriginObjective


class GaEvaluator:
    """
        pop_sizes           – list of population sizes to test
        crossover_methods   – list of crossover operators to test
        mutation_rates      – list of per-gene mutation probabilities
        selection_methods   – list of selection methods ('tournament'/'roulette')
        dim                 – problem dimensionality (fixed = 10)
        max_generations     – GA generations per run
        crossover_rate      – probability of crossover each pairing
        mutation_scale      – σ for Gaussian mutation
        runs                – number of independent runs per combo
        threshold           – fitness threshold for “poor” runs
        results             – dict mapping combo_key → metrics + avg_history
        df                  – DataFrame of scalar metrics (one row per combo)
    """

    def __init__(
        self,
        pop_sizes,
        crossover_methods,
        mutation_rates,
        selection_methods,
        runs=30,
        dim=10,
        max_generations=200,
        crossover_rate=0.9,
        mutation_scale=0.1,
        threshold=0.1
    ):
        self.pop_sizes = pop_sizes
        self.crossover_methods = crossover_methods
        self.mutation_rates = mutation_rates
        self.selection_methods = selection_methods
        self.runs = runs
        # Fixed GA/Rastrigin settings
        self.dim = dim
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_scale = mutation_scale

        # Flag runs whose final fitness < threshold
        self.threshold = threshold

        # To be filled after evaluate()
        self.results = {}
        self.df = None

    def _combo_key(self, pop, cross, mut, sel):
        return f"p{pop}_x{cross[:2]}_m{mut}_s{sel[:2]}"

    def evaluate(self):
        records = []
        combos = list(product(
            self.pop_sizes,
            self.crossover_methods,
            self.mutation_rates,
            self.selection_methods
        ))

        for pop, cross, mut, sel in tqdm(combos, desc="GA Hyperparam combos"):
            best_fits = []
            gen_of_bests = []
            histories = []

            for _ in range(self.runs):
                # build a fresh GA
                obj = RastriginObjective(dim=self.dim)
                fitness_fn = lambda x: 1.0 / (1.0 + obj.evaluate(x))

                ga = GeneticAlgorithm(
                    pop_size=pop,
                    gene_length=self.dim,
                    fitness_fn=fitness_fn,
                    representation='real',
                    bounds=obj.bounds,
                    max_generations=self.max_generations,
                    crossover_rate=self.crossover_rate,
                    mutation_rate=mut,
                    selection_method=sel,
                    tournament_size=3,
                    crossover_method=cross,
                    mutation_method='gaussian',
                    mutation_scale=self.mutation_scale
                )

                _, best_fit = ga.run()
                best_fits.append(best_fit)

                # generation of first attainment of best_fit
                gen_best = next(
                    (i+1 for i,v in enumerate(ga.history) if np.isclose(v, best_fit)),
                    self.max_generations
                )
                gen_of_bests.append(gen_best)
                histories.append(ga.history)

            # aggregate
            avg_fit = np.mean(best_fits)
            avg_gen = np.mean(gen_of_bests)
            avg_hist = np.mean(histories, axis=0)

            key = self._combo_key(pop, cross, mut, sel)
            self.results[key] = {
                'pop_size': pop,
                'crossover': cross,
                'mutation_rate': mut,
                'selection': sel,
                'avg_best_fitness': avg_fit,
                'avg_gen_of_best': avg_gen,
                'avg_history': avg_hist
            }

            records.append({
                'key': key,
                'pop_size': pop,
                'crossover': cross,
                'mutation_rate': mut,
                'selection': sel,
                'avg_best_fitness': avg_fit,
                'avg_gen_of_best': avg_gen
            })

        self.df = pd.DataFrame(records)
        return self.df, self.results

    def plot_hyperparameter_effects(self):
        """
        For each hyperparameter, plot mean ± std of final best fitness.
        """
        if self.df is None:
            raise RuntimeError("Call evaluate() first.")

        params = ['pop_size', 'mutation_rate', 'selection', 'crossover']
        fig, axes = plt.subplots(1, len(params), figsize=(20,5), sharey=True)

        for ax, p in zip(axes, params):
            grp = self.df.groupby(p)['avg_best_fitness']
            x = grp.mean().index
            y = grp.mean().values
            yerr = grp.std().values
            ax.errorbar(x, y, yerr=yerr, marker='o', linestyle='-')
            ax.set_title(f"Effect of {p}")
            ax.set_xlabel(p)
            ax.set_ylabel("Avg Best Fitness")
            ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_best_vs_worst(self):
        """
        Overlay convergence (avg_history) of the single best vs worst combos,
        chosen by avg_best_fitness ↑, then avg_gen_of_best ↓ as tie-breaker.
        """
        if self.df is None:
            raise RuntimeError("Call evaluate() first.")

        # best = max fitness, worst = min fitness
        sorted_df = self.df.sort_values(
            ['avg_best_fitness', 'avg_gen_of_best'],
            ascending=[False, True]
        )
        best = sorted_df.iloc[0]
        worst = sorted_df.iloc[-1]

        b = self.results[best['key']]
        w = self.results[worst['key']]

        plt.figure(figsize=(8,5))
        plt.plot(b['avg_history'],  label=f"BEST ({best['key']})", linewidth=2)
        plt.plot(w['avg_history'], '--', label=f"WORST ({worst['key']})", linewidth=2)
        plt.title("GA Convergence: Best vs. Worst Hyperparameters")
        plt.xlabel("Generation")
        plt.ylabel("Avg Best Fitness")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_top_bottom(self, n=5):
        """
        Overlay the top-n and bottom-n convergence curves in one figure.
        """
        if self.df is None:
            raise RuntimeError("Call evaluate() first.")

        # sort by fitness then gen_of_best
        sorted_df = self.df.sort_values(
            ['avg_best_fitness', 'avg_gen_of_best'],
            ascending=[False, True]
        )
        top_keys   = sorted_df['key'].head(n)
        bottom_keys= sorted_df['key'].tail(n)

        plt.figure(figsize=(8,5))
        for k in top_keys:
            plt.plot(self.results[k]['avg_history'],
                     label=f"TOP {k}", linewidth=1.5)
        for k in bottom_keys:
            plt.plot(self.results[k]['avg_history'],
                     '--', label=f"BOTTOM {k}", linewidth=1.5)
        plt.title(f"Top {n} vs Bottom {n} GA Convergence")
        plt.xlabel("Generation")
        plt.ylabel("Avg Best Fitness")
        plt.legend(fontsize='small', ncol=2)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()