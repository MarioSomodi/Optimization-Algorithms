import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from algorithms.simulated_annealing import SimulatedAnnealing
from objective_functions.rastrigin import RastriginObjective

class SimulatedAnnealingCoolingSchedulesEvaluator:
    """
    run and compare different SA cooling schedules on the Rastrigin objective function

    Attributes:
        schedules: Mapping of schedule names to their config params
            -linear
                alpha - linear temperature decrement per iteration
            -log
                no additional config needed
            -exponential
                beta - decay factor per iteration
            -adaptive
                window - number of recent moves to compute acceptance rate by
                rate_target - desired acceptance percent
                alpha_fast - fast temp adjust multiplier
                alpha_slow - slow temp adjust multiplier
            -custom
                alpha - linear decrement phase
                beta - exponential decay factor phase
                switch_point - iteration number to switch phase
        runs: how many times to run SA
        max_iter: how many times to run search inside SA
        tol: Convergence threshold; iteration when objective evaluation <= tol is recorded
        dim: Dimensionality of the Rastrigin problem.
        init_temp: Starting temperature T0 for SA
        step_size: Scale of each random perturbation

        results: Populated after run(); maps each schedule name to a dict with:
            best_vals: list of the best objective evaluation values per SA run,
            iters_to_tol: list of iterations needed to reach tol,
            accept_rates: list of accept/reject histories per SA run.
    """

    def __init__(
        self,
        schedules: dict = None,
        runs: int = 100,
        max_iter: int = 10000,
        tol: float = 1.0,
        dim: int = 8,
        init_temp: float = 100000,
        step_size: float = 0.1
    ):
        # Initialize cooling schedules with defaults if none provided
        self.schedules = schedules or {
            'linear': {'alpha': init_temp / max_iter},
            'log': {},
            'exponential': {'beta': 0.995},
            'adaptive': {
                'window': 200,
                'rate_target': 0.35,
                'alpha_fast': 0.9,
                'alpha_slow': 0.995
            },
            'custom': {
                'alpha': init_temp / (0.5 * max_iter),
                'beta': 0.99,
                'switch_point': max_iter // 2
            }
        }
        self.runs = runs
        self.max_iter = max_iter
        self.tol = tol
        self.dim = dim
        self.init_temp = init_temp
        self.step_size = step_size

        # placeholder for evaluation results
        self.results = {}

    @staticmethod
    def _run_single(args):
        """
        unpack args, run one SA evaluation, return metrics.
        """
        schedule, params, dim, init_temp, step_size, max_iter, tol = args
        obj = RastriginObjective(dim=dim)
        sa = SimulatedAnnealing(
            objective=obj,
            initial_temp=init_temp,
            schedule=schedule,
            schedule_params=params,
            step_size=step_size,
            max_iter=max_iter
        )
        sa.run()

        # best evaluation
        best_val = sa.best_eval
        # iterations to reach tolerance
        for i, val in enumerate(sa.history):
            if val <= tol:
                iters_to_tol = i
                break
        else:
            iters_to_tol = max_iter
        # acceptance history
        accept_history = sa.accept_history

        return schedule, best_val, iters_to_tol, accept_history

    def evaluate(self):
        """
        Execute the evaluation for each cooling schedule.
        """
        # tuple for each run of each schedule
        tasks = []
        for schedule, params in self.schedules.items():
            for _ in range(self.runs):
                tasks.append((
                    schedule,
                    params,
                    self.dim,
                    self.init_temp,
                    self.step_size,
                    self.max_iter,
                    self.tol
                ))

        # parallel run
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for out in tqdm(executor.map(self._run_single, tasks),
                            total=len(tasks),
                            desc="Evaluating"):
                results.append(out)

        # form results by schedule
        schedules_results = {schedule: {'best_vals': [], 'iters_to_tol': [], 'accept_rates': []}
               for schedule in self.schedules}
        for schedule, best_val, it_tol, accept_hist in results:
            schedules_results[schedule]['best_vals'].append(best_val)
            schedules_results[schedule]['iters_to_tol'].append(it_tol)
            schedules_results[schedule]['accept_rates'].append(accept_hist)

        self.results = schedules_results

    def plot(self):
        """
        Plot results:
          1) Boxplot of final best evaluations per schedule
          2) Bar chart of average iterations to reach tol
          3) Smoothed line plot of acceptance rates over iterations
        """
        if not self.results:
            raise RuntimeError("No data to plot. Run .evaluate() first.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Solution Quality
        axes[0].boxplot(
            [v['best_vals'] for v in self.results.values()],
            labels=self.results.keys()
        )
        axes[0].set_title("Solution Quality")
        axes[0].set_ylabel("Best Final Evaluation")
        axes[0].grid(True)

        # 2. Convergence Speed
        means = [np.mean(v['iters_to_tol']) for v in self.results.values()]
        axes[1].bar(self.results.keys(), means)
        axes[1].set_title("Convergence Speed")
        axes[1].set_ylabel("Avg. Iterations to Reach Tolerance")
        axes[1].grid(True)

        # 3. Exploration Rate
        for label, val in self.results.items():
            accept_array = np.array(val['accept_rates'])
            avg_accept = np.mean(accept_array, axis=0)
            smoothed = np.convolve(avg_accept, np.ones(5) / 5, mode='valid')
            axes[2].plot(smoothed, label=label)
        axes[2].set_title("Exploration Rate")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Acceptance Rate (Moving Avg)")
        axes[2].legend(fontsize=8)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()
