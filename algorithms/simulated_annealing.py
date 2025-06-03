"""
SimulatedAnnealing

A metaheuristic optimizer that mimics annealing in metallurgy:
• Starts at a high “temperature” allowing uphill moves to escape local minima.
• Gradually cools according to a chosen schedule to focus on exploitation.
• Tracks and returns the best solution found.

Cooling Schedules Supported:
  • linear      : T(k) = T0 − α·k
  • logarithmic : T(k) = T0 / log(k+1)
  • exponential : T(k) = T0 · β^k
  • adaptive    : adjust cooling based on recent acceptance rate
  • custom      : user-provided schedule function

Usage:
  sa = SimulatedAnnealing(
      objective=rastrigin_obj,
      initial_temp=100,
      schedule='exponential',
      schedule_params={'beta': 0.99},
      step_size=0.5,
      max_iter=10000
  )
  best_x, best_f = sa.run()
  summary = sa.summary()
"""

import numpy as np
import time

class SimulatedAnnealing:
    def __init__(
        self,
        objective,
        initial_temp: float,
        schedule: str = 'exponential',
        schedule_params: dict = None,
        step_size: float = 0.1,
        max_iter: int = 1000
    ):
        """
        objective        – an object with .bounds and .evaluate(x)
        initial_temp     – starting temperature T0
        schedule         – one of {'linear','log','exponential','adaptive','custom'}
        schedule_params  – parameters for the chosen schedule:
            • linear:      {'alpha': float}
            • logarithmic: {} (no extra params)
            • exponential: {'beta': float in (0,1)}
            • adaptive:    {'window': int, 'rate_target': float, 'alpha_fast': float, 'alpha_slow': float}
            • custom:      {'func': callable(k, T_prev) -> T_k}
        step_size        – standard deviation for Gaussian perturbation
        max_iter         – total number of annealing steps
        """
        self.obj = objective
        self.dim = len(objective.global_min)
        self.T0 = initial_temp
        self.schedule = schedule
        self.params = schedule_params or {}
        self.step_size = step_size
        self.max_iter = max_iter

        # State trackers
        self.best_solution = None
        self.best_eval = float('inf')
        self.current_solution = None
        self.current_eval = None
        self.history = []
        self.temps = []
        self.total_time = 0

        # For adaptive schedule
        if schedule == 'adaptive':
            cfg = self.params
            self.window = cfg.get('window', max_iter // 10)
            self.target_rate = cfg.get('rate_target', 0.2)
            self.alpha_fast = cfg.get('alpha_fast', 0.99)
            self.alpha_slow = cfg.get('alpha_slow', 0.999)
            self.accept_history = []

    def _schedule_temp(self, k, T_prev):
        """Compute temperature at iteration k based on selected schedule."""
        if self.schedule == 'linear':
            alpha = self.params.get('alpha', self.T0 / self.max_iter)
            return max(self.T0 - alpha * k, 1e-8)
        elif self.schedule == 'log':
            return self.T0 / np.log(k + 2)  # log(1) undefined, so offset
        elif self.schedule == 'exponential':
            beta = self.params.get('beta', 0.99)
            return self.T0 * (beta ** k)
        elif self.schedule == 'adaptive':
            # adjust alpha based on recent acceptance rate
            if k > 0 and len(self.accept_history) >= self.window:
                recent = self.accept_history[-self.window:]
                rate = sum(recent) / len(recent)
                # if we're accepting too many, cool faster; if too few, cool slower
                factor = self.alpha_fast if rate > self.target_rate else self.alpha_slow
                return T_prev * factor
            return T_prev * self.alpha_slow
        elif self.schedule == 'custom':
            func = self.params.get('func')
            if not callable(func):
                raise ValueError("Custom schedule requires a 'func' parameter.")
            return func(k, T_prev)
        else:
            raise ValueError(f"Unknown schedule '{self.schedule}'")

    def _perturb(self, x):
        """Generate a Gaussian‐noisy neighbor clipped to bounds."""
        candidate = x + np.random.normal(0, self.step_size, self.dim)
        low, high = self.obj.bounds
        return np.clip(candidate, low, high)

    def run(self):
        """Perform the simulated annealing search over max_iter steps."""
        start = time.perf_counter()
        # Initialize with noise near global min
        self.current_solution = self._perturb(self.obj.global_min)
        self.current_eval = self.obj.evaluate(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_eval = self.current_eval
        T = self.T0

        for k in range(self.max_iter):
            # Cooling: safely clamp temperature to avoid zero or negative
            T = max(self._schedule_temp(k, T), 1e-8)

            # Generate neighbor and evaluate
            neighbor = self._perturb(self.current_solution)
            neigh_eval = self.obj.evaluate(neighbor)

            # Compute change in energy
            delta = neigh_eval - self.current_eval

            # Acceptance logic with overflow safety
            if delta < 0:
                accept = True
            else:
                try:
                    accept_prob = np.exp(-min(delta / T, 700))  # avoid overflow
                    accept = np.random.rand() < accept_prob
                except FloatingPointError:
                    accept = False

            # Apply move if accepted
            if accept:
                self.current_solution = neighbor
                self.current_eval = neigh_eval
                accepted = 1
                if neigh_eval < self.best_eval:
                    self.best_solution = neighbor.copy()
                    self.best_eval = neigh_eval
            else:
                accepted = 0

            # Logging
            self.history.append(self.current_eval)
            self.temps.append(T)
            if self.schedule == 'adaptive':
                self.accept_history.append(accepted)

        self.total_time = time.perf_counter() - start
        return self.best_solution, self.best_eval


    def summary(self):
        """
        Returns:
          best_solution  – best point found
          best_evaluation– its objective value
          total_time     – wall-clock duration
          history        – list of accepted current_eval per iteration
          temps          – temperature schedule over iterations
        """
        return {
            'best_solution': self.best_solution,
            'best_evaluation': self.best_eval,
            'total_time': self.total_time,
            'history': self.history,
            'temps': self.temps
        }
