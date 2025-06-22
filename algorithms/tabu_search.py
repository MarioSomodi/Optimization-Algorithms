"""
Tabu Search is a metaheuristic optimization algorithm that guides a local search
procedure to explore the solution space beyond local optimality by using memory structures.

The algorithm iteratively explores the neighborhood of the current solution,
selecting the best candidate that is not forbidden by the tabu list. The tabu list
keeps track of recently visited moves (or attributes of moves) to prevent cycles
and encourage exploration. Aspiration criteria allow overriding the tabu status
if a move yields a globally best solution.
"""
import numpy as np
from collections import deque

class TabuSearch:
    """
    objective:    object with .evaluate(x) and .bounds
    max_iters:    total number of iterations
    tabu_tenure:  length of the tabu list
    neighborhood_size: number of neighbors to sample each iter
    step_size:    coordinate perturbation magnitude
    """
    def __init__(self, objective, max_iters=1000,
                 tabu_tenure=10, neighborhood_size=20, step_size=0.1):
        self.obj = objective
        self.bounds = np.array([objective.bounds] * len(objective.global_min))
        self.dim = len(objective.global_min)
        self.max_iters = max_iters
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size
        self.step_size = step_size

        # Will be filled after .run()
        self.best_solution = None
        self.best_eval = None
        self.history = []

    def _random_neighbor(self, x):
        """Generate one neighbor by ±step on a random coordinate."""
        i = np.random.randint(self.dim)
        direction = np.random.choice([-1, 1])
        x_new = x.copy()
        x_new[i] += direction * self.step_size
        low, high = self.bounds[i]
        x_new[i] = np.clip(x_new[i], low, high)
        move = (i, direction)
        return x_new, move

    def run(self):
        x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        best = x.copy()
        best_val = self.obj.evaluate(x)
        tabu = deque(maxlen=self.tabu_tenure)
        history = [best_val]

        for _ in range(self.max_iters):
            candidates = []
            for _ in range(self.neighborhood_size):
                x_cand, move = self._random_neighbor(x)
                f_cand = self.obj.evaluate(x_cand)
                if move in tabu and f_cand >= best_val:
                    continue
                candidates.append((x_cand, f_cand, move))

            if not candidates:
                break

            # select best candidate
            x_cand, val_cand, move = min(candidates, key=lambda t: t[1])

            # aspiration: allow tabu move if it's better
            if move in tabu and val_cand >= best_val:
                for cand in sorted(candidates, key=lambda t: t[1]):
                    if cand[2] not in tabu:
                        x_cand, val_cand, move = cand
                        break

            x = x_cand
            tabu.append(move)
            if val_cand < best_val:
                best, best_val = x.copy(), val_cand

            history.append(best_val)

        # Store results
        self.best_solution = best
        self.best_eval = best_val
        self.history = history

        return best, best_val, history
