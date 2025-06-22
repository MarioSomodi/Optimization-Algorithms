import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from algorithms.tabu_search import TabuSearch
from objective_functions.rastrigin import RastriginObjective

class TsEvaluator:
    """
        tabu_tenures: List of tabu list lengths to test.
        step_sizes: List of coordinate step sizes to test.
        neighborhood_sizes: List of neighborhood sizes to test.
        dim: Dimensionality of the Rastrigin objective function.
        runs: Number of independent runs per configuration.
        max_iters: Maximum number of iterations per Tabu Search run.
        threshold: Convergence threshold for f(x); used to determine success.
    """
    def __init__(self,
                 tabu_tenures=[5, 10, 20],
                 step_sizes=[0.05, 0.1, 0.2],
                 neighborhood_sizes=[10, 20],
                 dim=2,
                 runs=40,
                 max_iters=10000,
                 threshold=1.0):
        self.tabu_tenures = tabu_tenures
        self.step_sizes = step_sizes
        self.neighborhood_sizes = neighborhood_sizes
        self.dim = dim
        self.runs = runs
        self.max_iters = max_iters
        self.threshold = threshold

        self.df = None

    @staticmethod
    def _run_single(args):
        T, s, k, dim, max_iters, threshold = args
        obj = RastriginObjective(dim)
        ts = TabuSearch(
            objective=obj,
            max_iters=max_iters,
            tabu_tenure=T,
            neighborhood_size=k,
            step_size=s
        )
        _, val, hist = ts.run()
        conv_it = next((i for i, v in enumerate(hist) if v < threshold), np.nan)
        return T, s, k, val, conv_it

    def evaluate(self):
        tasks = [
            (T, s, k, self.dim, self.max_iters, self.threshold)
            for T in self.tabu_tenures
            for s in self.step_sizes
            for k in self.neighborhood_sizes
            for _ in range(self.runs)
        ]

        results = []
        with ProcessPoolExecutor() as executor:
            for out in tqdm(executor.map(self._run_single, tasks),
                            total=len(tasks),
                            desc="Evaluating"):
                results.append(out)

        records = []
        grouped = {}
        for T, s, k, val, conv_it in results:
            key = (T, s, k)
            grouped.setdefault(key, {'vals': [], 'iters': []})
            grouped[key]['vals'].append(val)
            grouped[key]['iters'].append(conv_it)

        for (T, s, k), d in grouped.items():
            finite_iters = [it for it in d['iters'] if np.isfinite(it)]
            records.append({
                'tabu_tenure': T,
                'step_size': s,
                'neigh_size': k,
                'avg_final': np.mean(d['vals']),
                'std_final': np.std(d['vals']),
                'conv_rate': len(finite_iters) / self.runs,
                'avg_conv_it': np.mean(finite_iters) if finite_iters else np.nan
            })

        self.df = pd.DataFrame(records)

    def plot(self):
        if self.df is None:
            raise RuntimeError("Call evaluate() before plotting.")

        for metric, title, label, cmap, vmin, vmax in [
            ('avg_final', 'Avg Final Fitness', 'Avg Final f', 'YlGnBu', None, None),
            ('conv_rate', 'Convergence Rate', 'Rate', 'Greens', 0, 1)
        ]:
            subset = self.df[self.df['neigh_size'] == max(self.neighborhood_sizes)]
            pivot = subset.pivot(index='tabu_tenure', columns='step_size', values=metric)
            extent = [-0.5, len(self.step_sizes) - 0.5, -0.5, len(self.tabu_tenures) - 0.5]
            xticks = range(len(self.step_sizes))
            yticks = range(len(self.tabu_tenures))

            plt.figure(figsize=(6, 5))
            plt.title(f"Tabu Search {title}\n(neigh_size={max(self.neighborhood_sizes)}) (convergance threshold={self.threshold})", fontsize=14, weight='bold')
            im = plt.imshow(pivot.values, origin='lower', extent=extent,
                            aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.xticks(xticks, self.step_sizes, fontsize=10)
            plt.yticks(yticks, self.tabu_tenures, fontsize=10)
            plt.xlabel('step_size', fontsize=12)
            plt.ylabel('tabu_tenure', fontsize=12)

            # Annotate each cell with its value
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.iloc[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=9)

            # Highlight the best config (min final, max rate)
            if metric == 'avg_final':
                best_idx = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
            elif metric == 'conv_rate':
                best_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.shape)

            plt.gca().add_patch(plt.Rectangle(
                (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                fill=False, edgecolor='red', linewidth=2.0))

            plt.colorbar(im, label=label)
            plt.grid(False)
            plt.tight_layout()
            plt.show()