import numpy as np

class AntColonyOptimization:
    """
    Ant Colony Optimization for the Traveling Salesman Problem (TSP).

    Args:
        distance_matrix: 2D numpy array of pairwise distances between cities.
        n_ants: Number of ants per iteration.
        n_iterations: Number of colony iterations.
        decay: Pheromone evaporation coefficient in [0,1].
        alpha: Influence of pheromone on decision.
        beta: Influence of heuristic (inverse distance) on decision.
    """
    def __init__(
        self,
        distance_matrix,
        n_ants=10,
        n_iterations=100,
        decay=0.5,
        alpha=1.0,
        beta=2.0
    ):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        # Initialize pheromone levels
        self.pheromone = np.ones((self.num_cities, self.num_cities))
        # Heuristic info: inverse of distance (avoid div by zero on diagonal)
        self.heuristic = 1.0 / (distance_matrix + np.eye(self.num_cities))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        # Best‐so‐far tracking
        self.best_route = None
        self.best_distance = float('inf')
        self.history = []  # stores best_distance at each iteration

    def _route_length(self, route):
        """Compute total tour length (closing the loop)."""
        total = 0.0
        for i in range(len(route)-1):
            total += self.distance_matrix[route[i], route[i+1]]
        total += self.distance_matrix[route[-1], route[0]]
        return total

    def _select_next_city(self, current, visited):
        """Probability‐based city selection."""
        tau = self.pheromone[current] ** self.alpha
        eta = self.heuristic[current] ** self.beta
        mask = np.ones(self.num_cities, bool)
        mask[visited] = False
        probs = tau[mask] * eta[mask]
        probs /= probs.sum()
        choices = np.where(mask)[0]
        return np.random.choice(choices, p=probs)

    def _construct_solutions(self):
        """Each ant builds a complete tour."""
        all_routes, all_dists = [], []
        for _ in range(self.n_ants):
            tour = [np.random.randint(self.num_cities)]
            while len(tour) < self.num_cities:
                tour.append(self._select_next_city(tour[-1], tour))
            all_routes.append(tour)
            all_dists.append(self._route_length(tour))
        return all_routes, all_dists

    def _update_pheromone(self, routes, dists):
        """Evaporate and deposit pheromone."""
        # evaporation
        self.pheromone *= (1 - self.decay)
        # deposit proportional to 1/distance
        for tour, dist in zip(routes, dists):
            deposit = 1.0 / dist
            for i in range(len(tour)-1):
                a, b = tour[i], tour[i+1]
                self.pheromone[a,b] += deposit
                self.pheromone[b,a] += deposit
            # close loop
            a, b = tour[-1], tour[0]
            self.pheromone[a,b] += deposit
            self.pheromone[b,a] += deposit

    def run(self):
        """
        Run the ACO optimization without per-iteration printing.
        Returns: (best_route, best_distance)
        """
        for _ in range(self.n_iterations):
            routes, dists = self._construct_solutions()
            self._update_pheromone(routes, dists)

            curr_best = min(dists)
            if curr_best < self.best_distance:
                self.best_distance = curr_best
                self.best_route = routes[np.argmin(dists)]
            self.history.append(self.best_distance)
        return self.best_route, self.best_distance

    def summary(self):
        """Return a summary of the optimization results and parameters."""
        return {
            'n_ants': self.n_ants,
            'n_iterations': self.n_iterations,
            'decay': self.decay,
            'alpha': self.alpha,
            'beta': self.beta,
            'best_route': self.best_route,
            'best_distance': self.best_distance,
            'history': list(self.history),
            'pheromone_matrix': self.pheromone.copy()
        }