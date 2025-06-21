# objective_functions/rosenbrock.py
import numpy as np
from objective_functions.base_objective import BaseObjective

class RosenbrockObjective(BaseObjective):
    def __init__(self, dim: int):
        self.dim = dim
        super().__init__(
            name="Rosenbrock",
            bounds=(-2.048, 2.048),
            global_min=np.ones(dim)
        )

    def evaluate(self, x):
        # f(x) = Σ[100·(x_{i+1} − x_i^2)^2 + (1−x_i)^2]
        return sum(
            100.0*(x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
            for i in range(self.dim - 1)
        )
