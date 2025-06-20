import matplotlib.pyplot as plt

def quadratic_objective_function(x):
    """
    Example objective function: f(x) = (x - 3)^2 + 1
    """
    return (x - 3)**2 + 1


def numerical_gradient(f, x, h=1e-5):
    """
    Estimate the gradient of function f at point x using central differences.

    Args:
        f: Objective function taking a scalar x.
        x: Point at which to estimate the gradient.
        h: Small step size for finite differences.

    Returns:
        Approximate derivative f'(x).
    """
    # Central difference formula
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_descent(f, starting_point, learning_rate, num_iterations):
    """
    Perform basic gradient descent to minimize function f.

    Args:
        f: Objective function.
        starting_point: Initial guess for x.
        learning_rate: Step size multiplier.
        num_iterations: Number of iterations to perform.

    Returns:
        x: Estimated location of the minimum.
        f_history: List of function values at each iteration.
        x_history: List of x values at each iteration.
    """
    x = starting_point
    x_history = []  # store x values for plotting
    f_history = []  # store f(x) values for plotting

    for i in range(num_iterations):
        # Estimate gradient at current x
        grad = numerical_gradient(f, x)
        # Record current state
        x_history.append(x)
        f_history.append(f(x))
        # Update rule: move in the opposite direction of the gradient
        x = x - learning_rate * grad

    # Record final state
    x_history.append(x)
    f_history.append(f(x))

    return x, f_history, x_history