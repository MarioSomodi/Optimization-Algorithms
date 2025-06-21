import numpy as np
from collections import deque

class TabuSearch:
    def __init__(self, func, bounds, max_iters=1000,
                 tabu_tenure=10, neighborhood_size=20, step_size=0.1):
        """
        func:        callable f(x: np.ndarray) -> float
        bounds:      list of (low, high) for each dim
        max_iters:   total number of iterations
        tabu_tenure: length of the tabu list
        neighborhood_size: how many neighbors to sample each iter
        step_size:   coordinate perturbation magnitude
        """
        self.f = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.max_iters = max_iters
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size
        self.step_size = step_size

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
        # initialize
        x = np.random.uniform(self.bounds[:,0], self.bounds[:,1])
        best, best_val = x.copy(), self.f(x)
        tabu = deque(maxlen=self.tabu_tenure)
        history = [best_val]

        for it in range(self.max_iters):
            # sample neighbors
            candidates = []
            for _ in range(self.neighborhood_size):
                x_cand, move = self._random_neighbor(x)
                if move in tabu and self.f(x_cand) >= best_val:
                    continue
                candidates.append((x_cand, self.f(x_cand), move))

            if not candidates:
                break

            # pick best candidate
            x_cand, val_cand, move = min(candidates, key=lambda t: t[1])
            # aspiration: allow tabu if it sets new global best
            if move in tabu and val_cand >= best_val:
                # pick next best non-tabu
                for cand in sorted(candidates, key=lambda t: t[1]):
                    if cand[2] not in tabu:
                        x_cand, val_cand, move = cand
                        break

            # move
            x = x_cand
            tabu.append(move)
            if val_cand < best_val:
                best, best_val = x.copy(), val_cand

            history.append(best_val)

        return best, best_val, history