"""
Nelder-Mead is a simplex-based optimization algorithm for unconstrained problems.
    It maintains a geometric figure (simplex) of n+1 vertices in n dimensions and
    iteratively reflects, expands, contracts, or shrinks the simplex to move toward
    a local minimum of the objective function.
    Unlike gradient-based methods, it uses only function evaluations (no derivatives),
    making it suitable for noisy or non-smooth functions.
"""
import numpy as np

class NelderMead:
    """
        objective:  objective function object with .evaluate(x) and .bounds
        x0:         initial point (1D numpy array)
        max_iters:  maximum number of iterations
        tol:        tolerance for convergence (stddev of f-values)
        alpha:      reflection coefficient
        gamma:      expansion coefficient
        rho:        contraction coefficient
        sigma:      shrink coefficient
    """
    def __init__(self, objective, x0, max_iters=200, tol=1e-6,
                 alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.obj = objective
        self.f = objective.evaluate
        self.dim = len(x0)
        self.max_iters = max_iters
        self.tol = tol
        self.alpha, self.gamma, self.rho, self.sigma = alpha, gamma, rho, sigma

        # build initial simplex
        self.simplex = np.vstack([
            x0,
            *(x0 + np.eye(self.dim)[i] * 0.05 for i in range(self.dim))
        ])

        self.best_solution = None
        self.best_eval = None
        self.history = []

    def run(self):
        fvals = np.apply_along_axis(self.f, 1, self.simplex)
        self.history = [np.min(fvals)]

        for it in range(self.max_iters):
            # sort simplex and evaluate
            idx = np.argsort(fvals)
            self.simplex = self.simplex[idx]
            fvals = fvals[idx]

            best, worst = self.simplex[0], self.simplex[-1]
            centroid = np.mean(self.simplex[:-1], axis=0)

            # reflection
            xr = centroid + self.alpha * (centroid - worst)
            fr = self.f(xr)

            if fvals[0] <= fr < fvals[-2]:
                self.simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[0]:
                # expansion
                xe = centroid + self.gamma * (xr - centroid)
                fe = self.f(xe)
                if fe < fr:
                    self.simplex[-1], fvals[-1] = xe, fe
                else:
                    self.simplex[-1], fvals[-1] = xr, fr
            else:
                # contraction
                xc = centroid + self.rho * (worst - centroid)
                fc = self.f(xc)
                if fc < fvals[-1]:
                    self.simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink
                    self.simplex = best + self.sigma * (self.simplex - best)
                    fvals = np.apply_along_axis(self.f, 1, self.simplex)

            self.history.append(np.min(fvals))

        idx = np.argmin(fvals)
        self.best_solution = self.simplex[idx]
        self.best_eval = fvals[idx]