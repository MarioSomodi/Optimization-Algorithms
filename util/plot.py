import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(history,
    title="Cumulative Best Convergence",
    label="Best value so far"):
    best_so_far = np.minimum.accumulate(history)
    plt.figure(figsize=(10,6))
    plt.plot(best_so_far, color='r', marker='o', label=label)
    plt.xlabel('Iteration')
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_total_time_vs_mn(mn_product, total_times):
    plt.figure(figsize=(8, 5))
    plt.plot(mn_product, total_times, color='r', marker='o', label='Total Time (s)')
    plt.xlabel('m·n (Function Evaluations)')
    plt.ylabel('Total Time (s)')
    plt.title('Total Time vs. m·n')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_growth_rate(mn, total_times):
    growth_rates = [(total_times[i] - total_times[i-1])/(mn[i] - mn[i-1])
                    for i in range(1, len(total_times))]
    plt.figure(figsize=(8,5))
    plt.plot(mn[1:], growth_rates, color='r', marker='o', label='Growth Rate (s per eval)')
    plt.xlabel('m·n (Function Evaluations)')
    plt.ylabel('Growth Rate (s per eval)')
    plt.title('Growth Rate of Runtime per Evaluation')
    plt.grid(True)
    plt.legend()
    plt.show()