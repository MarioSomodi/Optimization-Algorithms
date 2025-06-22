import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.gradient_descent import GradientDescent
from objective_functions.quadratic import QuadraticObjective

class GdEvaluator:
    """
    Runs GD with both exact and numerical gradients on QuadraticObjective.

    Attributes:
        starting_point – initial vector (array-like)
        h_values – list of finite difference step sizes for numerical gradient
        learning_rates – list of learning rates η
        num_iterations – GD steps per run
        results – list of dicts with method, history, final values
        df – summary DataFrame of all runs
    """
    def __init__(self,
                 starting_point=[5.0, 5.0],
                 h_values=None,
                 learning_rates=None,
                 num_iterations=20):
        self.starting_point = np.array(starting_point, dtype=float)
        self.h_values = h_values or [1e-1, 1e-3, 1e-5, 1e-8]
        self.learning_rates = learning_rates or [0.001, 0.01, 0.1, 0.5]
        self.num_iterations = num_iterations

        self.results = []
        self.df = None
        self.objective = QuadraticObjective()

    def run(self):
        self.results.clear()
        for lr in self.learning_rates:
            # Run with exact gradient (analytical)
            gd_exact = GradientDescent(
                objective=self.objective,
                starting_point=self.starting_point,
                learning_rate=lr,
                num_iterations=self.num_iterations,
                h=1e-5
            )
            gd_exact.run()
            self.results.append({
                'method': 'exact',
                'h': np.nan,
                'lr': lr,
                'x_history': gd_exact.x_history,
                'f_history': gd_exact.f_history,
                'x_final': gd_exact.final_x,
                'f_final': gd_exact.final_f
            })

            # Numerical gradient with each h
            for h in self.h_values:
                gd_numerical = GradientDescent(
                    objective=self.objective,
                    starting_point=self.starting_point,
                    learning_rate=lr,
                    num_iterations=self.num_iterations,
                    h=h
                )
                gd_numerical.run()
                self.results.append({
                    'method': 'numerical',
                    'h': h,
                    'lr': lr,
                    'x_history': gd_numerical.x_history,
                    'f_history': gd_numerical.f_history,
                    'x_final': gd_numerical.final_x,
                    'f_final': gd_numerical.final_f
                })

        # Build summary DataFrame
        rows = [{
            'method': r['method'],
            'h': r['h'],
            'lr': r['lr'],
            'x_final': r['x_final'],
            'f_final': r['f_final']
        } for r in self.results]

        self.df = pd.DataFrame(rows)

    def plot_convergence(self):
        if not self.results:
            raise RuntimeError("Call run() first.")

        for method in ['exact', 'numerical']:
            plt.figure(figsize=(8, 5))
            subset = [r for r in self.results if r['method'] == method]
            for r in subset:
                label = f"lr={r['lr']}" + ("" if method == 'exact' else f", h={r['h']}")
                plt.plot(r['f_history'], marker='o', label=label)
            plt.title(f"Convergence using {method} Gradient")
            plt.xlabel("Iteration")
            plt.ylabel("f(x)")
            plt.grid(True)
            plt.legend(fontsize='small')
            plt.tight_layout()
            plt.show()

    def plot_path(self, lr=0.1, h=1e-5, method='numerical'):
        match = next(
            (r for r in self.results
             if r['method'] == method and
                np.isclose(r['lr'], lr) and
                (method == 'exact' or np.isclose(r['h'], h))),
            None
        )
        if match is None:
            raise ValueError("No run found for those parameters.")

        xs = np.linspace(0, 6, 200)
        ys = [self.objective.evaluate([x, x]) for x in xs]

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, label='f(x) diagonal slice')
        fs = match['f_history']
        plt.plot([x[0] for x in match['x_history']], fs, '-o', label='Path')
        plt.title(f"Path (lr={lr}, h={h if method == 'numerical' else '—'})")
        plt.xlabel("x[0] (diagonal slice)")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
