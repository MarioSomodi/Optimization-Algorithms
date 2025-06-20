import numpy as np

class AntColonyOptimization:
    """
    Ant Colony Optimization for the Traveling Salesman Problem (TSP).

    Args:
        distance_matrix: 2D numpy array of pairwise distances between cities.
        n_ants: Number of ants per iteration.
        n_iterations: Number of colony iterations.
        decay: Pheromone evaporation coefficient in [0,1]. (ρ)
        alpha: Influence of pheromone on decision. (α)
        beta: Influence of heuristic (inverse distance) on decision. (β)
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
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay  # evaporation rate ρ
        self.alpha = alpha  # pheromone importance α
        self.beta = beta    # heuristic importance β

        # Initialize pheromone matrix τ_{ij} = 1
        # Formula: τ_{ij}(0) = 1
        self.pheromone = np.ones((self.num_cities, self.num_cities))

        # Initialize heuristic matrix η_{ij} = 1 / d_{ij}
        # Formula: η_{ij} = 1 / d_{ij}, with η_{ii} undefined (set via +I)
        self.heuristic = 1.0 / (distance_matrix + np.eye(self.num_cities))

        # Best solution tracking
        self.best_route = None
        self.best_distance = float('inf')
        self.history = []

    def _route_length(self, route):
        """
        Compute total length of a tour.
        Formula: L = Σ_{i=0 to n-2} d_{route[i],route[i+1]} + d_{route[-1],route[0]}
        """
        total = 0.0
        for i in range(len(route)-1):
            total += self.distance_matrix[route[i], route[i+1]]
        total += self.distance_matrix[route[-1], route[0]]  # return to start
        return total

    def _select_next_city(self, current, visited):
        """
        Probabilistic selection of the next city.
        Formula: P_{ij} = (τ_{ij}^α * η_{ij}^β) / Σ_{k∉visited}(τ_{ik}^α * η_{ik}^β)
        """
        # Compute pheromone^α and heuristic^β
        tau = self.pheromone[current] ** self.alpha
        eta = self.heuristic[current] ** self.beta

        # Mask visited cities
        mask = np.ones(self.num_cities, bool)
        mask[visited] = False

        # Calculate selection probabilities
        probs = tau[mask] * eta[mask]
        probs /= probs.sum()

        choices = np.where(mask)[0]
        return np.random.choice(choices, p=probs)

    def _construct_solutions(self):
        """
        Build tours for all ants.
        Each ant starts at a random city and selects next cities via P_{ij} until all visited.
        """
        all_routes, all_dists = [], []
        for _ in range(self.n_ants):
            # Random start
            tour = [np.random.randint(self.num_cities)]
            # Grow tour
            while len(tour) < self.num_cities:
                tour.append(self._select_next_city(tour[-1], tour))
            all_routes.append(tour)
            all_dists.append(self._route_length(tour))
        return all_routes, all_dists

    def _update_pheromone(self, routes, dists):
        """
        Update pheromone trails after all ants have built tours.
        Evaporation: τ_{ij} ← (1 - ρ) * τ_{ij}
        Deposition: τ_{ij} ← τ_{ij} + Σ_{k}(Δτ_{ij}^k), where Δτ_{ij}^k = Q / L_k (Q=1)
        """
        # Evaporation
        self.pheromone *= (1 - self.decay)

        # Deposition
        for tour, dist in zip(routes, dists):
            deposit = 1.0 / dist  # Δτ = Q / L
            # Update each edge in the tour
            for i in range(len(tour)-1):
                a, b = tour[i], tour[i+1]
                self.pheromone[a, b] += deposit
                self.pheromone[b, a] += deposit
            # Close loop
            a, b = tour[-1], tour[0]
            self.pheromone[a, b] += deposit
            self.pheromone[b, a] += deposit

    def run(self):
        """
        iterate solution construction and pheromone update.
        """
        for _ in range(self.n_iterations):
            routes, dists = self._construct_solutions()
            self._update_pheromone(routes, dists)

            # Track best solution
            curr_best = min(dists)
            if curr_best < self.best_distance:
                self.best_distance = curr_best
                self.best_route = routes[np.argmin(dists)]
            self.history.append(self.best_distance)

        return self.best_route, self.best_distance

    def summary(self):
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
