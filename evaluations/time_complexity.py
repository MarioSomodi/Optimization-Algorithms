import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""
evaluate_time_complexity

function to measure how an algorithms runtime increases
with the total number of function evaluations (max_iter·local_iter).  

1.Instantiate the objective and optimizer with those parameters.
   • Run the optimizer once, capturing:
       – Total run time
       – Average time per outer restart
       – Average time per inner local-search loop
   • Record m, n, m·n, and all timing metrics.
2. Return both:
   • raw_data: Python lists of each metric (for custom plotting or analysis)
   • df: a pandas DataFrame formatted for easy display and export
"""

def evaluate_time_complexity(algo, objective_class, config, runs=6,
                              max_iter_start=10, local_search_start=5,
                              max_iter_step=10, local_search_step=5,
                              dim=2, step_size=0.5):
    max_iters = []
    local_iters = []
    total_times = []
    outer_times = []
    local_times = []
    mn_product = []

    for i in range(runs):
        max_iter = max_iter_start + i * max_iter_step
        local_iter = local_search_start + i * local_search_step

        obj = objective_class(dim=dim)
        algorithm = algo(
            objective=obj,
            step_size=step_size,
            max_iter=max_iter,
            local_search_iter=local_iter,
            **config
        )

        algorithm.run()
        summary = algorithm.summary()

        max_iters.append(max_iter)
        local_iters.append(local_iter)
        outer_times.append(summary['avg_outer_time'])
        local_times.append(summary['avg_local_time'])
        total_times.append(summary['total_time'])
        mn_product.append(max_iter * local_iter)

    raw_data = {
        'max_iters': max_iters,
        'local_iters': local_iters,
        'total_times': total_times,
        'outer_times': outer_times,
        'local_times': local_times,
        'mn_product': mn_product
    }
    df = pd.DataFrame({
        'input_size_max_iter': max_iters,
        'input_size_local_search': local_iters,
        'm·n': mn_product,
        'total_time (s)': total_times,
        'time_max_iter (s)': np.array(outer_times) - np.array(local_times),
        'time_local_search (s)': local_times,
    })

    return raw_data, df

def plot_runtime_analysis_inline(mn_product, total_times):
    """
    Plot two graphs:
    1. Total time vs. total function evaluations (m·n), which reflects empirical time complexity.
    2. Growth rate of time per evaluation, to analyze if cost per step increases.

    Inputs:
    - mn_product: list of m·n values where:
        m = max_iter (max iter)
        n = local_iter (local search iter)
    - total_times: corresponding total run times for each m·n point
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Shows how runtime grows with number of function evaluations
    axes[0].plot(mn_product, total_times, color='r', marker='o', markersize='2', label='Total Time (s)')
    axes[0].set_title("Total Time vs. m·n")
    axes[0].set_xlabel("m·n (Function Evaluations)")
    axes[0].set_ylabel("Total Time (s)")
    axes[0].legend()
    axes[0].grid(True)

    # how much time increases per additional evaluation
    growth_rates = [(total_times[i] - total_times[i-1]) / (mn_product[i] - mn_product[i-1])
                    for i in range(1, len(total_times))]
    axes[1].plot(mn_product[1:], growth_rates, color='r', marker='o', markersize='2', label='Growth Rate (s per eval)')
    axes[1].set_title("Growth Rate of Runtime per Evaluation")
    axes[1].set_xlabel("m·n (Function Evaluations)")
    axes[1].set_ylabel("Growth Rate (s per eval)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()