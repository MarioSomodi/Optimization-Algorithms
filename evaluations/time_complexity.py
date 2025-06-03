import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

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

    
    print("outer_times:", outer_times)
    print("local_times:", local_times)
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