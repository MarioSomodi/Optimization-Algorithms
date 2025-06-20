import numpy as np
import matplotlib.pyplot as plt

# Task 16: Experiment with Basic Gradient Descent

def quadratic_objective_function(x):
    """
    Quadratic objective: f(x) = (x - 3)^2 + 1
    """
    return (x - 3)**2 + 1

def analytical_derivative(x):
    """
    Exact gradient of f(x).
    f'(x) = 2*(x - 3)
    """
    return 2 * (x - 3)

def numerical_gradient(f, x, h=1e-5):
    """
    Numerical estimate of gradient via central differences.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient_descent(f, grad_fn, starting_point, learning_rate, num_iterations):
    """
    Performs gradient descent given a gradient function.

    Returns final x, f_history, x_history.
    """
    x = starting_point
    x_history = [x]
    f_history = [f(x)]
    for i in range(num_iterations):
        grad = grad_fn(x)
        x = x - learning_rate * grad
        x_history.append(x)
        f_history.append(f(x))
    return x, f_history, x_history

def run_experiments():
    starting_point = 5.0
    h_values = [1e-1, 1e-3, 1e-5, 1e-8, 1e-10]
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    num_iterations = 20

    results = []
    # Loop over parameters
    for h in h_values:
        for lr in learning_rates:
            # Exact gradient
            x_e, f_e, xh_e = gradient_descent(
                quadratic_objective_function,
                analytical_derivative,
                starting_point,
                lr,
                num_iterations
            )
            results.append({
                'method': 'exact', 'h': None, 'lr': lr,
                'x_final': x_e, 'f_final': f_e[-1],
                'f_history': f_e, 'x_history': xh_e
            })

            # Numerical gradient
            grad_num = lambda x_val: numerical_gradient(
                quadratic_objective_function, x_val, h
            )
            x_n, f_n, xh_n = gradient_descent(
                quadratic_objective_function,
                grad_num,
                starting_point,
                lr,
                num_iterations
            )
            results.append({
                'method': 'numerical', 'h': h, 'lr': lr,
                'x_final': x_n, 'f_final': f_n[-1],
                'f_history': f_n, 'x_history': xh_n
            })

    # Visualization: f_history convergence
    for method in ['exact', 'numerical']:
        plt.figure(figsize=(10, 6))
        for res in [r for r in results if r['method'] == method]:
            label = f"lr={res['lr']}" + (
                "" if method == 'exact' else f", h={res['h']}"
            )
            plt.plot(res['f_history'], marker='o', label=label)
        plt.title(f"Convergence of f(x) using {method.capitalize()} Gradient")
        plt.xlabel("Iteration")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Visualization: optimization path for one case
    # Choose numerical with h=1e-5 and lr=0.1 as example
    selected = next(
        r for r in results
        if r['method'] == 'numerical' and r['h'] == 1e-5 and r['lr'] == 0.1
    )
    xs = np.linspace(0, 6, 200)
    ys = quadratic_objective_function(xs)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, label='f(x)')
    plt.plot(
        selected['x_history'], selected['f_history'],
        marker='o', linestyle='--', label='Optimization Path'
    )
    plt.title('Optimization Path (Numerical: h=1e-5, lr=0.1)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()