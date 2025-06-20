import numpy as np

class GeneticAlgorithm:
    """
    A simple Genetic Algorithm implementation supporting both continuous (real-valued)
    and discrete (binary) representations.
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
        self.mutation_method = mutation_method or ('gaussian' if representation == 'real' else 'flip_bit')
        self.mutation_scale = mutation_scale

        # Initialize population
        if self.representation == 'real':
            low, high = self.bounds
            self.population = np.random.uniform(low, high, (pop_size, gene_length))
        else:
            self.population = np.random.randint(0, 2, (pop_size, gene_length))

    def evaluate_population(self):
        return np.array([self.fitness_fn(ind) for ind in self.population])

    def tournament_selection(self, fitnesses):
        idx = np.random.choice(self.pop_size, self.tournament_size, replace=False)
        return self.population[idx[np.argmax(fitnesses[idx])]].copy()

    def roulette_wheel_selection(self, fitnesses):
        total_fit = fitnesses.sum()
        if total_fit == 0:
            return self.population[np.random.randint(self.pop_size)].copy()
        probs = fitnesses / total_fit
        return self.population[np.random.choice(self.pop_size, p=probs)].copy()

    def select_parent(self, fitnesses):
        return (self.tournament_selection if self.selection_method == 'tournament'
                else self.roulette_wheel_selection)(fitnesses)

    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        if self.crossover_method == 'single_point':
            pt = np.random.randint(1, self.gene_length)
            return (np.concatenate([parent1[:pt], parent2[pt:]]),
                    np.concatenate([parent2[:pt], parent1[pt:]]))
        if self.crossover_method == 'two_point':
            p1, p2 = sorted(np.random.choice(range(1, self.gene_length), 2, replace=False))
            return (np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]]),
                    np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]]))
        if self.crossover_method == 'uniform':
            mask = np.random.rand(self.gene_length) < 0.5
            return (np.where(mask, parent1, parent2),
                    np.where(mask, parent2, parent1))
        raise ValueError(f"Unknown crossover: {self.crossover_method}")

    def mutate(self, ind):
        if self.representation == 'real' and self.mutation_method == 'gaussian':
            for i in range(self.gene_length):
                if np.random.rand() < self.mutation_rate:
                    ind[i] += np.random.normal(0, self.mutation_scale)
            low, high = self.bounds
            np.clip(ind, low, high, out=ind)
        elif self.representation == 'binary' and self.mutation_method == 'flip_bit':
            for i in range(self.gene_length):
                if np.random.rand() < self.mutation_rate:
                    ind[i] = 1 - ind[i]
        else:
            raise ValueError(f"Unknown mutation: {self.mutation_method}")
        return ind

    def run(self):
        best_ind, best_fit = None, -np.inf
        for _ in range(self.max_generations):
            fits = self.evaluate_population()
            idx = np.argmax(fits)
            if fits[idx] > best_fit:
                best_fit, best_ind = fits[idx], self.population[idx].copy()
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.select_parent(fits)
                p2 = self.select_parent(fits)
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))
            self.population = np.array(new_pop)
        return best_ind, best_fit

# --- Helper functions to run GA examples ---

def run_ga_rastrigin(
    dim=10,
    pop_size=100,
    max_generations=200,
    crossover_rate=0.9,
    mutation_rate=0.05,
    selection_method='tournament',
    tournament_size=3,
    crossover_method='uniform',
    mutation_scale=0.1
):
    from objective_functions.rastrigin import RastriginObjective
    obj = RastriginObjective(dim=dim)
    fitness_fn = lambda x: 1.0 / (1.0 + obj.evaluate(x))
    ga = GeneticAlgorithm(
        pop_size=pop_size,
        gene_length=dim,
        fitness_fn=fitness_fn,
        representation='real',
        bounds=obj.bounds,
        max_generations=max_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_method=selection_method,
        tournament_size=tournament_size,
        crossover_method=crossover_method,
        mutation_method='gaussian',
        mutation_scale=mutation_scale
    )
    best_sol, best_fit = ga.run()
    return best_sol, obj.evaluate(best_sol)


def run_ga_knapsack(
    knapsack_items,
    capacity=100,
    pop_size=100,
    max_generations=200,
    crossover_rate=0.8,
    mutation_rate=0.02,
    selection_method='roulette',
    crossover_method='two_point'
):
    gene_length = len(knapsack_items)
    def fitness(ind):
        total_val = sum(item['value'] for bit, item in zip(ind, knapsack_items) if bit)
        total_wt = sum(item['weight'] for bit, item in zip(ind, knapsack_items) if bit)
        return total_val if total_wt <= capacity else 0
    ga = GeneticAlgorithm(
        pop_size=pop_size,
        gene_length=gene_length,
        fitness_fn=fitness,
        representation='binary',
        bounds=None,
        max_generations=max_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_method=selection_method,
        crossover_method=crossover_method,
        mutation_method='flip_bit'
    )
    best_sol, best_fit = ga.run()
    return best_sol, best_fit