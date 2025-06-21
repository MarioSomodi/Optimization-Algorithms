import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

class TimeComplexityEvaluator:
    """
    Measures how an algorithm's runtime scales with the total number of function evaluations.

    Attributes:
        algo: Optimizer class to instantiate.
        objective_class: Objective function class to instantiate.
        config: Extra config passed to the optimizer.
        runs: Number of runs at increasing sizes.
        max_iter_start, local_search_start, max_iter_step, local_search_step:
            Parameters to generate the series of (max_iter, local_iter) pairs.
        dim: Dimensionality for the objective.
        step_size: Step size passed to the optimizer.
        raw_data: Collected lists of metrics after evaluate().
        df: Tabular view of the collected metrics.
    """
    def __init__(
        self,
        algo,
        objective_class,
        config: dict,
        runs: int = 6,
        max_iter_start: int = 10,
        local_search_start: int = 5,
        max_iter_step: int = 10,
        local_search_step: int = 5,
        dim: int = 2,
        step_size: float = 0.5
    ):
        self.algo = algo
        self.objective_class = objective_class
        self.config = config
        self.runs = runs
        self.max_iter_start = max_iter_start
        self.local_search_start = local_search_start
        self.max_iter_step = max_iter_step
        self.local_search_step = local_search_step
        self.dim = dim
        self.step_size = step_size

        self.raw_data = {}
        self.df: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _run_single(args):
        """
        unpack args, run one timing experiment, return raw stats.
        """
        algo, objective_class, config, dim, step_size, m, n = args
        obj = objective_class(dim=dim)
        optimizer = algo(
            objective=obj,
            step_size=step_size,
            max_iter=m,
            local_search_iter=n,
            **config
        )
        optimizer.run()
        stats = optimizer.get_timing_data()
        return {
            'm': m,
            'n': n,
            'total_time': stats['total_time'],
            'avg_outer_time': stats['avg_outer_time'],
            'avg_local_time': stats['avg_local_time']
        }

    def evaluate(self):
        """
        Run all experiments in parallel and store:
          - raw_data: dict of lists
          - df: pandas.DataFrame
        """
        # prepare m,n pairs
        pairs = [
            (
                self.max_iter_start + i * self.max_iter_step,
                self.local_search_start + i * self.local_search_step
            )
            for i in range(self.runs)
        ]

        tasks = [
            (
                self.algo,
                self.objective_class,
                self.config,
                self.dim,
                self.step_size,
                m, n
            )
            for m, n in pairs
        ]

        #run parallel
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for res in tqdm(executor.map(self._run_single, tasks),
                            total=len(tasks),
                            desc="Evaluating"):
                results.append(res)

        # unpack
        max_iters = [r['m'] for r in results]
        local_iters = [r['n'] for r in results]
        total_times = [r['total_time'] for r in results]
        outer_times = [r['avg_outer_time'] for r in results]
        local_times = [r['avg_local_time'] for r in results]
        mn_product = [m*n for m,n in zip(max_iters, local_iters)]

        self.raw_data = {
            'max_iters': max_iters,
            'local_iters': local_iters,
            'm·n': mn_product,
            'total_times': total_times,
            'outer_times': outer_times,
            'local_times': local_times
        }

        self.df = pd.DataFrame({
            'input_size_max_iter': max_iters,
            'input_size_local_search': local_iters,
            'm·n': mn_product,
            'total_time (s)': total_times,
            'time_max_iter (s)': np.array(outer_times) - np.array(local_times),
            'time_local_search (s)': local_times,
        })

    def plot(self):
        """
        Plot:
          1) Total time vs. m·n
          2) Growth rate per evaluation
        """
        if not self.raw_data:
            raise RuntimeError("No data to plot. Run .evaluate() first.")

        mn = self.raw_data['m·n']
        total = self.raw_data['total_times']
        growth = [
            (total[i] - total[i-1]) / (mn[i] - mn[i-1])
            for i in range(1, len(total))
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(mn, total, marker='o', label='Total Time (s)')
        axes[0].set_title("Total Time vs. m·n")
        axes[0].set_xlabel("m·n (Function Evaluations)")
        axes[0].set_ylabel("Total Time (s)")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(mn[1:], growth, marker='o', label='Growth Rate (s per eval)')
        axes[1].set_title("Growth Rate per Evaluation")
        axes[1].set_xlabel("m·n (Function Evaluations)")
        axes[1].set_ylabel("Growth Rate (s per eval)")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()
