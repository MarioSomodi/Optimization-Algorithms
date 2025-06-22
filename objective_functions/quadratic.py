import numpy as np
from objective_functions.base_objective import BaseObjective

class QuadraticObjective(BaseObjective):
    """
    General n‐dimensional quadratic objective (default 2‐D):
        f(x) = ∑_{i=1}^d w_i · (x_i − c_i)^2 + constant
      dim       – dimensionality d (default 2)
      center    – list/array of length d giving each c_i (defaults to all 3’s)
      weights   – list/array of length d giving each w_i (defaults to all 1’s)
      constant  – scalar offset (defaults to +1)
    """
    def __init__(self, dim=2, center=None, weights=None, constant=1.0):
        # if no center provided, default to all 3's; if provided as scalar, broadcast
        if center is None:
            self.center = np.full(dim, 3.0)
        else:
            self.center = np.atleast_1d(center).astype(float)
            # if user passed a scalar center, broadcast it to length dim
            if self.center.size == 1 and dim > 1:
                self.center = np.full(dim, self.center.item())
        # same for weights
        if weights is None:
            self.weights = np.ones(dim)
        else:
            self.weights = np.atleast_1d(weights).astype(float)
            if self.weights.size == 1 and dim > 1:
                self.weights = np.full(dim, self.weights.item())

        # ensure center/weights length matches dim
        assert self.center.size == dim, "center length must match dim"
        assert self.weights.size == dim, "weights length must match dim"

        super().__init__(
            name=f"Quadratic_{dim}D",
            bounds=([-np.inf]*dim, [np.inf]*dim),
            global_min=self.center.copy()
        )
        self.constant = float(constant)

    def evaluate(self, x):
        """
        x : array-like of length d
        """
        xv = np.asarray(x, dtype=float)
        diff = xv - self.center
        return np.sum(self.weights * diff**2) + self.constant
