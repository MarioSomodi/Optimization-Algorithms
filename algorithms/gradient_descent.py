import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    """
      objective – BaseObjective with method evaluate(x) → float
      starting_point – initial guess array-like (length d)
      learning_rate – step size η
      num_iterations – how many updates to perform
      h – finite-difference step size

      x_history - solution history
      f_history – eval history
      final_x – last x
      final_f – last eval
    """
    def __init__(self, objective, starting_point, learning_rate=0.1, num_iterations=100, h=1e-5):
        self.obj = objective
        self.x0 = np.array(starting_point, dtype=float)
        self.eta = learning_rate
        self.iters = num_iterations
        self.h = h

        self.x_history = []
        self.f_history = []
        self.final_x = None
        self.final_f = None

    def _numerical_gradient(self, x):
        """Estimate ∇f(x) by central differences, one coordinate at a time."""
        d = x.size
        grad = np.zeros(d, dtype=float)
        for i in range(d):
            x_ph = x.copy()
            x_ph[i] += self.h
            x_mh = x.copy()
            x_mh[i] -= self.h
            grad[i] = (self.obj.evaluate(x_ph) - self.obj.evaluate(x_mh)) / (2 * self.h)
        return grad

    def run(self):
        """
        Perform gradient descent.
        """
        x = self.x0.copy()
        self.x_history = [x.copy()]
        self.f_history = [self.obj.evaluate(x)]

        for _ in range(self.iters):
            grad = self._numerical_gradient(x)
            x = x - self.eta * grad
            self.x_history.append(x.copy())
            self.f_history.append(self.obj.evaluate(x))

        self.final_x = x
        self.final_f = self.obj.evaluate(x)

    def plot_convergence(self):
        """
        Plot objective vs iteration, and if d==2 plot the trajectory in the plane.
        """
        if not self.f_history:
            raise RuntimeError("Call run() before plot_convergence()")

        iters = np.arange(len(self.f_history))
        fig = plt.figure(figsize=(12,4))

        # 1) objective vs iteration
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(iters, self.f_history, marker='o')
        ax1.set_title("f(x) vs. iteration")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("f(x)")
        ax1.grid(True)

        # 2) if 2-D, show path
        start = self.x_history[0]
        if start.size == 2:

            ax2 = fig.add_subplot(1, 2, 2)
            xs = [p[0] for p in self.x_history]
            ys = [p[1] for p in self.x_history]
            ax2.plot(xs, ys, '-o')
            ax2.set_title("Descent path in 2 dim")
            ax2.set_xlabel("x₀")
            ax2.set_ylabel("x₁")
            ax2.grid(True)

        plt.tight_layout()
        plt.show()
