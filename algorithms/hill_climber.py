"""
HillClimber

1. Repeat max_iter times:
   • Pick a fresh random point in the bounds.
   • Perform step_size steps: 
       - Propose a neighbor by using random distribution.
       - If the neighbor is better, move there.
       - Always remember the single best solution seen across all restarts.
2. At the end, returns the best solution found and its score.
3. Tracks timing and final scores per restart for plots.
"""

import numpy as np
import time

class HillClimber:
    def __init__(self, objective, step_size=0.1, max_iter=1000, local_search_iter=500):
        """
        objective an object with .bounds (low,high) and .evaluate(x) methods
        step_size standard deviation of normal distribution values
        max_iter how many restarts to run
        local_search_iter how many steps per local search
        """
        self.obj = objective
        self.dim = len(objective.global_min)
        self.step_size = step_size
        self.max_iter = max_iter
        self.local_search_iter = local_search_iter

        # placeholders for tracking results and timing
        self.best_solution = None      
        self.best_eval = float('inf')  
        self.history = []              
        self.local_times = []          
        self.outer_times = []          
        self.total_time = 0            

    def _random_start(self):
        """Pick a new random point in the bounds."""
        low, high = self.obj.bounds
        return np.random.uniform(low, high, self.dim)

    def _generate_neighbor(self, x):
        """Propose a small neighbor in the normal distribution, clipped to stay in bounds."""
        candidate = x + np.random.normal(0, self.step_size, self.dim)
        low, high = self.obj.bounds
        return np.clip(candidate, low, high)

    def run(self):
        """Execute max_iter restarts of local searches for local_search_iter times, tracking the best overall score."""
        start_time = time.perf_counter()

        for _ in range(self.max_iter):
            outer_start = time.perf_counter()
            # begin from a fresh random point
            current = self._random_start()
            current_eval = self.obj.evaluate(current)

            local_start = time.perf_counter()
            # local search: only accept improving neighbors
            for _ in range(self.local_search_iter):
                neighbor = self._generate_neighbor(current)
                neighbor_eval = self.obj.evaluate(neighbor)

                if neighbor_eval < current_eval:
                    # move to the better neighbor
                    current, current_eval = neighbor, neighbor_eval

                    # update global best if this is best so far
                    if current_eval < self.best_eval:
                        self.best_solution = current
                        self.best_eval = current_eval

            # record timings and final score for this restart
            self.local_times.append(time.perf_counter() - local_start)
            self.outer_times.append(time.perf_counter() - outer_start)
            self.history.append(current_eval)

        self.total_time = time.perf_counter() - start_time
        return self.best_solution, self.best_eval

    def summary(self):
        """
        Return a dictionary with:
          • best_solution the best point found
          • best_evaluation its score
          • total_time full run duration
          • avg_local_time average inner loop time per restart
          • avg_outer_time average full restart time
          • history list of final scores per restart
        """
        return {
            'best_solution': self.best_solution,
            'best_evaluation': self.best_eval,
            'total_time': self.total_time,
            'avg_local_time': np.mean(self.local_times),
            'avg_outer_time': np.mean(self.outer_times),
            'history': self.history
        }