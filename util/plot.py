import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(history,
    title="Convergence",
    label="Best value"):
    best_so_far = np.minimum.accumulate(history)
    plt.figure(figsize=(10,6))
    plt.plot(best_so_far, color='r', marker='o', markersize='2', label=label)
    plt.xlabel('Iteration')
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()