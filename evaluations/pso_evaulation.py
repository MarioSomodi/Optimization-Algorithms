import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from objective_functions.rastrigin import RastriginObjective

class PsoEvaluator:
    """
        c1_values: the list of cognitive coefficients to test
        c2_values: the list of social coefficients to test
        w_values: the list of inertia weights to test
        runs: number of independent PSO runs per parameter combination
        max_iter: maximum iterations per PSO run
        tol: convergence threshold; record iteration when best eval <= tol
        dim: dimensionality of the Rastrigin problem
        n_particles: number of particles in the swarm

        results: filled after evaluate()
        df: pandas DataFrame view of results for easy analysis and plotting
    """
    def __init__(
        self,
        c1_values,
        c2_values,
        w_values,
        runs=30,
        max_iter=100,
        tol=1e-3,
        dim=2,
        n_particles=30
    ):
        self.c1_values = c1_values
        self.c2_values = c2_values
        self.w_values = w_values
        self.runs = runs
        self.max_iter = max_iter
        self.tol = tol
        self.dim = dim
        self.n_particles = n_particles
        self.results = {}
        self.df = None

    @staticmethod
    def _run_single(args):
        """
        unpack args, run one PSO experiment, return metrics.
        """
        w, c1, c2, dim, n_particles, max_iter, tol = args
        obj = RastriginObjective(dim)
        pso = ParticleSwarmOptimization(
            objective=obj,
            n_particles=n_particles,
            w=w,
            c1=c1,
            c2=c2,
            max_iter=max_iter
        )
        pso.run()

        best_eval = pso.gbest_score
        total_time = pso.total_time
        history = np.array(pso.history)

        # iteration when best eval first <= tol
        reached = np.where(history <= tol)[0]
        iter_to_tol = int(reached[0]) if reached.size > 0 else max_iter

        return w, c1, c2, best_eval, total_time, iter_to_tol, history

    def evaluate(self):
        """
        Run PSO for each combination of w, c1, and c2,
        aggregate performance metrics and convergence histories.
        """
        # prepare task arguments for all runs
        tasks = []
        for w in self.w_values:
            for c1 in self.c1_values:
                for c2 in self.c2_values:
                    for _ in range(self.runs):
                        tasks.append((
                            w, c1, c2,
                            self.dim,
                            self.n_particles,
                            self.max_iter,
                            self.tol
                        ))

        # parallel execution
        total_runs = len(tasks)
        results_list = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for out in tqdm(executor.map(self._run_single, tasks),
                            total=total_runs,
                            desc="Evaluating"):
                results_list.append(out)

        #format by param variations
        param_variations = {}
        for w, c1, c2, global_best_score, total_time, iter_to_tol, history in results_list:
            key = f"w{w}_c1{c1}_c2{c2}"
            if key not in param_variations:
                param_variations[key] = {
                    'w': w,
                    'c1': c1,
                    'c2': c2,
                    'global_best_scores': [],
                    'total_times': [],
                    'iters_to_tol': [],
                    'histories': []
                }
            entry = param_variations[key]
            entry['global_best_scores'].append(global_best_score)
            entry['total_times'].append(total_time)
            entry['iters_to_tol'].append(iter_to_tol)
            entry['histories'].append(history)

        # compute averages and build df
        records = []
        for key, entry in param_variations.items():
            avg_global_best_score = np.mean(entry['global_best_scores'])
            avg_total_time = np.mean(entry['total_times'])
            avg_iter_to_tol = np.mean(entry['iters_to_tol'])
            avg_history = np.mean(np.vstack(entry['histories']), axis=0)

            self.results[key] = {
                'w': entry['w'],
                'c1': entry['c1'],
                'c2': entry['c2'],
                'avg_global_best_score': avg_global_best_score,
                'avg_total_time': avg_total_time,
                'avg_iter_to_tol': avg_iter_to_tol,
                'avg_history': avg_history
            }
            rec = {'key': key,
                   'w': entry['w'],
                   'c1': entry['c1'],
                   'c2': entry['c2'],
                   'avg_global_best_score': avg_global_best_score,
                   'avg_total_time': avg_total_time,
                   'avg_iter_to_tol': avg_iter_to_tol}
            records.append(rec)

        self.df = pd.DataFrame(records)

    def plot(self):
        """
        Plot mean convergence curves for each inertia weight in separate subplots.
        """
        if self.df is None:
            raise RuntimeError("No data to plot. Call .evaluate() first.")

        w_vals = sorted(self.df['w'].unique())
        n_w = len(w_vals)
        fig, axes = plt.subplots(1, n_w, figsize=(6 * n_w, 5), sharex=True, sharey=True)
        if n_w == 1:
            axes = [axes]

        for ax, w in zip(axes, w_vals):
            subset = self.df[self.df['w'] == w]
            for _, row in subset.iterrows():
                label = f"c1={row['c1']}, c2={row['c2']}"
                ax.plot(
                    self.results[row['key']]['avg_history'],
                    alpha=0.7,
                    label=label
                )
            ax.set_title(f"PSO Convergence (w={w})")
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best-so-far Eval')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize='small', loc='upper right')

        plt.tight_layout()
        plt.show()

    def top_bottom(self, n=5):
        if self.df is None:
            raise RuntimeError("No data: run .evaluate() first.")

        sorted_best = self.df.sort_values(
            ['avg_global_best_score', 'avg_iter_to_tol', 'avg_total_time'],
            ascending=[True, True, True]
        )
        top_keys = sorted_best['key'].head(n).tolist()

        sorted_worst = self.df.sort_values(
            ['avg_global_best_score', 'avg_iter_to_tol', 'avg_total_time'],
            ascending=[False, False, False]
        )
        worst_keys = sorted_worst['key'].head(n).tolist()

        return top_keys, worst_keys


    def plot_top_bottom(self, n=5):
        """
        Overlay the average convergence curves of the top-n and worst-n
        hyperparameter combos on a single graph.
        """
        if self.df is None:
            raise RuntimeError("No data: run .evaluate() first.")

        top_keys, worst_keys = self.top_bottom(n)

        plt.figure(figsize=(8, 5))
        # plot best n
        for key in top_keys:
            meta = self.results[key]
            label = f"BEST {key} (score={meta['avg_global_best_score']:.3g})"
            plt.plot(meta['avg_history'], label=label, linewidth=1.5)

        # plot worst n
        for key in worst_keys:
            meta = self.results[key]
            label = f"WORST {key} (score={meta['avg_global_best_score']:.3g})"
            plt.plot(meta['avg_history'], linestyle='--', label=label, linewidth=1.5)

        plt.title(f"Top {n} vs Worst {n} PSO Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Best‐so‐far Evaluation")
        plt.legend(fontsize='small', ncol=2)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_best_vs_worst(self):
        """
        Plot the convergence curve of the single best and single worst
        parameter combinations on one graph by
          1) avg_global_best_score
          2) avg_iter_to_tol
          3) avg_total_time
        """
        if self.df is None:
            raise RuntimeError("No data to plot. Run .evaluate() first.")

        sorted_best = self.df.sort_values(
            ['avg_global_best_score', 'avg_iter_to_tol', 'avg_total_time'],
            ascending=[True, True, True]
        )
        best_row = sorted_best.iloc[0]

        sorted_worst = self.df.sort_values(
            ['avg_global_best_score', 'avg_iter_to_tol', 'avg_total_time'],
            ascending=[False, False, False]
        )
        worst_row = sorted_worst.iloc[0]

        best_key  = best_row['key']
        worst_key = worst_row['key']

        best_hist  = self.results[best_key]['avg_history']
        worst_hist = self.results[worst_key]['avg_history']

        b = self.results[best_key]
        w = self.results[worst_key]
        best_label  = (f"Best: w={b['w']}, c1={b['c1']}, c2={b['c2']}\n"
                       f"score={best_row['avg_global_best_score']:.4g}, "
                       f"iters={best_row['avg_iter_to_tol']:.0f}, "
                       f"time={best_row['avg_total_time']:.3f}s")
        worst_label = (f"Worst: w={w['w']}, c1={w['c1']}, c2={w['c2']}\n"
                       f"score={worst_row['avg_global_best_score']:.4g}, "
                       f"iters={worst_row['avg_iter_to_tol']:.0f}, "
                       f"time={worst_row['avg_total_time']:.3f}s")

        plt.figure(figsize=(8,5))
        plt.plot(best_hist,  label=best_label,  linewidth=2)
        plt.plot(worst_hist, label=worst_label, linestyle='--', linewidth=2)
        plt.title("PSO Convergence: Best vs Worst Configurations")
        plt.xlabel("Iteration")
        plt.ylabel("Best‐so‐far Evaluation")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

