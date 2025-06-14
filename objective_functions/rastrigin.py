import numpy as np
from objective_functions.base_objective import BaseObjective


class RastriginObjective(BaseObjective):
    def __init__(self, dim: int, a: float = 10):
        self.a = a
        self.dim = dim
        super().__init__(name="Rastrigin", bounds=(-5.12, 5.12), global_min=np.zeros(dim))

    def evaluate(self, x):
        """
        Rastrigin function: f(x) = An + Σ[x_i² - A * cos(2πx_i)]
        Global minimum at x = 0^n, value = 0
        """
        return self.a * self.dim + sum(x_i ** 2 - self.a * np.cos(2 * np.pi * x_i) for x_i in x)