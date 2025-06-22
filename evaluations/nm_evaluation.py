import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from algorithms.nelder_mead import NelderMead
from objective_functions.rosenbrock import RosenbrockObjective

class NmEvaluator:
    def __init__(self,
                 alphas=[0.8, 1.0, 1.2],
                 gammas=[1.5, 2.0],
                 rhos=[0.3, 0.5],
                 sigmas=[0.3, 0.5],
                 dim=2,
                 runs=20,
                 max_iters=500,
                 threshold=1e-6):
        self.alphas = alphas
        self.gammas = gammas
        self.rhos = rhos
        self.sigmas = sigmas
        self.dim = dim
        self.runs = runs
        self.max_iters = max_iters
        self.threshold = threshold
        self.df = None

    @staticmethod
    def _run_single(args):
        alpha, gamma, rho, sigma, dim, max_iters, threshold = args
        obj = RosenbrockObjective(dim)
        nm = NelderMead(
            objective=obj,
            x0=np.zeros(dim),
            max_iters=max_iters,
            tol=threshold,
            alpha=alpha,
            gamma=gamma,
            rho=rho,
            sigma=sigma
        )
        nm.run()
        final_val = nm.best_eval
        history = nm.history

        if final_val < threshold:
            conv_it = next((i for i, v in enumerate(history) if v < threshold), np.nan)
        else:
            conv_it = np.nan

        return alpha, gamma, rho, sigma, nm.best_eval, conv_it

    def evaluate(self):
        tasks = [
            (a, g, r, s, self.dim, self.max_iters, self.threshold)
            for a in self.alphas
            for g in self.gammas
            for r in self.rhos
            for s in self.sigmas
            for _ in range(self.runs)
        ]

        results = []
        with ProcessPoolExecutor() as executor:
            for out in tqdm(executor.map(self._run_single, tasks),
                            total=len(tasks),
                            desc="Evaluating"):
                results.append(out)

        grouped = {}
        for a, g, r, s, val, conv_it in results:
            key = (a, g, r, s)
            grouped.setdefault(key, {'vals': [], 'iters': []})
            grouped[key]['vals'].append(val)
            grouped[key]['iters'].append(conv_it)

        records = []
        for (a, g, r, s), d in grouped.items():
            finite = [i for i in d['iters'] if np.isfinite(i)]
            records.append({
                'alpha': a,
                'gamma': g,
                'rho': r,
                'sigma': s,
                'avg_final': np.mean(d['vals']),
                'std_final': np.std(d['vals']),
                'conv_rate': len(finite) / self.runs,
                'avg_conv_it': np.mean(finite) if finite else np.nan
            })

        self.df = pd.DataFrame(records)

    def plot(self):
        if self.df is None:
            raise RuntimeError("Call evaluate() before plotting.")

        fixed_df = self.df[(self.df['rho'] == 0.5) & (self.df['sigma'] == 0.5)].copy()
        if fixed_df.empty:
            raise ValueError("No entries found with rho=0.5 and sigma=0.5")

        pivot_avg = fixed_df.pivot(index='alpha', columns='gamma', values='avg_final')
        pivot_rate = fixed_df.pivot(index='alpha', columns='gamma', values='conv_rate')

        extent = [-0.5, len(self.gammas) - 0.5, -0.5, len(self.alphas) - 0.5]
        xticks = range(len(self.gammas))
        yticks = range(len(self.alphas))

        plots = [
            (pivot_avg, 'Avg Final Function Value', 'Avg f(x)', 'YlGnBu', '.6f', None),
            (pivot_rate, 'Convergence Rate (f < threshold)', 'Rate', 'BuGn', '.0%', 1)
        ]

        for pivot, title, label, cmap, fmt, vmax in plots:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_title(f"Nelder–Mead: {title}\n(rho=0.5, sigma=0.5)", fontsize=14, weight='bold')

            im = ax.imshow(pivot.values, cmap=cmap, origin='lower', extent=extent,
                           aspect='auto', vmin=0, vmax=vmax)

            ax.set_xticks(xticks)
            ax.set_xticklabels(self.gammas)
            ax.set_yticks(yticks)
            ax.set_yticklabels(self.alphas)
            ax.set_xlabel('gamma', fontsize=12)
            ax.set_ylabel('alpha', fontsize=12)

            # Annotate each cell with its value
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.iloc[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=9)

            # Highlight the best cell (min final, max rate)
            best_idx = (np.nanargmin(pivot.values) if label == 'Avg f(x)'
                        else np.nanargmax(pivot.values))
            best_idx = np.unravel_index(best_idx, pivot.shape)
            ax.add_patch(plt.Rectangle(
                (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                fill=False, edgecolor='red', linewidth=2.0
            ))

            fig.colorbar(im, ax=ax, label=label)
            plt.tight_layout()
            plt.show()
