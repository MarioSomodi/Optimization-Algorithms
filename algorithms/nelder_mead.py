import numpy as np

class NelderMead:
    def __init__(self, func, x0, max_iters=200, tol=1e-6,
                 alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """
        func:       callable f(x: np.ndarray) -> float
        x0:         initial point (1d array)
        tol:        stopping tolerance on simplex size
        alpha, γ, ρ, σ: NM coefficients
        """
        self.f = func
        self.dim = len(x0)
        self.max_iters = max_iters
        self.tol = tol
        self.alpha, self.gamma, self.rho, self.sigma = alpha, gamma, rho, sigma

        # build initial simplex
        self.simplex = np.vstack([
            x0,
            *(x0 + np.eye(self.dim)[i] * 0.05 for i in range(self.dim))
        ])

    def run(self):
        fvals = np.apply_along_axis(self.f, 1, self.simplex)
        history = [np.min(fvals)]

        for it in range(self.max_iters):
            # sort simplex
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

            history.append(np.min(fvals))

            # check convergence
            if np.std(fvals) < self.tol:
                break

        idx = np.argmin(fvals)
        return self.simplex[idx], fvals[idx], history
