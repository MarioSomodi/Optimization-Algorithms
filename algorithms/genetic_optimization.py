"""
population-based metaheuristic that evolves a set of candidate solutions
over successive generations using selection, crossover, and mutation.
  1. Initialize a population of individuals (real-valued vectors or binary strings).
  2. Evaluate each individual’s fitness via a supplied fitness function.
  3. Select parents based on fitness (tournament or roulette wheel).
  4. Crossover parents to produce offspring (single-point, two-point, or uniform).
  5. Mutate offspring with some probability (Gaussian perturbation or a bit flip).
  6. Repeat for a fixed number of generations, tracking the best solution found.
"""

import numpy as np

class GeneticAlgorithm:
    """
        pop_size – number of individuals in the population
        gene_length - number of genes per individual (problem dimensionality)
        fitness_fn – function(individual) → scalar fitness (higher is better)
        representation – 'real' for continuous genes or 'binary' for bitstrings
        bounds – tuple (low, high) for real-valued genes; ignored for binary
        max_generations – how many generations to evolve
        crossover_rate – probability of performing crossover on a parent pair
        mutation_rate – probability of mutating each gene in an offspring
        selection_method – 'tournament' or 'roulette' wheel selection
        tournament_size – number of competitors in each tournament (if tournament selection)
        crossover_method – 'single_point', 'two_point', or 'uniform' crossover
        mutation_method – 'gaussian' for real or 'flip_bit' for binary; defaults appropriately
        mutation_scale – standard deviation of Gaussian noise (for real mutation only)
    """
    def __init__(
        self,
        pop_size,
        gene_length,
        fitness_fn,
        representation='real',
        bounds=None,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.01,
        selection_method='tournament',
        tournament_size=3,
        crossover_method='single_point',
        mutation_method=None,
        mutation_scale=0.1
    ):
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.fitness_fn = fitness_fn
        self.representation = representation
        self.bounds = bounds
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method or (
            'gaussian' if representation == 'real' else 'flip_bit'
        )
        self.mutation_scale = mutation_scale
        self.history = []

        # initialize population
        if representation == 'real':
            low, high = self.bounds
            # uniform random real values in [low, high]
            self.population = np.random.uniform(low, high, (pop_size, gene_length))
        else:
            # random binary strings
            self.population = np.random.randint(0, 2, (pop_size, gene_length))

    def evaluate_population(self):
        """Evaluate fitness for each individual in the population."""
        return np.array([self.fitness_fn(ind) for ind in self.population])

    def tournament_selection(self, fitnesses):
        """Pick the best of a random subset (tournament) of individuals."""
        idx = np.random.choice(self.pop_size, self.tournament_size, replace=False)
        best = idx[np.argmax(fitnesses[idx])]
        return self.population[best].copy()

    def roulette_wheel_selection(self, fitnesses):
        """Select one individual with probability proportional to fitness."""
        total = fitnesses.sum()
        if total == 0:
            # fallback to random pick if all fitness are zero
            return self.population[np.random.randint(self.pop_size)].copy()
        probs = fitnesses / total
        i = np.random.choice(self.pop_size, p=probs)
        return self.population[i].copy()

    def select_parent(self, fitnesses):
        """Wrapper to choose selection method."""
        if self.selection_method == 'tournament':
            return self.tournament_selection(fitnesses)
        else:
            return self.roulette_wheel_selection(fitnesses)

    def crossover(self, parent1, parent2):
        """Combine two parents to produce two offspring."""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.crossover_method == 'single_point':
            # cut point in [1, gene_length-1)
            pt = np.random.randint(1, self.gene_length)
            c1 = np.concatenate([parent1[:pt], parent2[pt:]])
            c2 = np.concatenate([parent2[:pt], parent1[pt:]])
            return c1, c2

        if self.crossover_method == 'two_point':
            # two cut points
            p1, p2 = sorted(np.random.choice(
                range(1, self.gene_length), 2, replace=False))
            c1 = np.concatenate([
                parent1[:p1],
                parent2[p1:p2],
                parent1[p2:]
            ])
            c2 = np.concatenate([
                parent2[:p1],
                parent1[p1:p2],
                parent2[p2:]
            ])
            return c1, c2

        if self.crossover_method == 'uniform':
            # each gene has 50/50 chance of coming from either parent
            mask = np.random.rand(self.gene_length) < 0.5
            c1 = np.where(mask, parent1, parent2)
            c2 = np.where(mask, parent2, parent1)
            return c1, c2

        raise ValueError(f"Unknown crossover method: {self.crossover_method}")

    def mutate(self, individual):
        """Apply mutation to an individual in place."""
        if self.representation == 'real' and self.mutation_method == 'gaussian':
            # add Gaussian noise to each gene with probability mutation_rate
            for i in range(self.gene_length):
                if np.random.rand() < self.mutation_rate:
                    individual[i] += np.random.normal(0, self.mutation_scale)
            # enforce bounds
            low, high = self.bounds
            np.clip(individual, low, high, out=individual)

        elif self.representation == 'binary' and self.mutation_method == 'flip_bit':
            # flip each bit with probability mutation_rate
            for i in range(self.gene_length):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = 1 - individual[i]

        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")

        return individual

    def run(self):
        """
        Execute the GA for max_generations. Returns the best individual and its fitness.
        """
        best_ind, best_fit = None, -np.inf
        self.history = []

        for gen in range(self.max_generations):
            fitnesses = self.evaluate_population()

            # update global best
            idx = np.argmax(fitnesses)
            if fitnesses[idx] > best_fit:
                best_fit = fitnesses[idx]
                best_ind = self.population[idx].copy()

            self.history.append(best_fit)

            # create next generation
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.select_parent(fitnesses)
                p2 = self.select_parent(fitnesses)
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))

            self.population = np.array(new_pop)

        return best_ind, best_fit
