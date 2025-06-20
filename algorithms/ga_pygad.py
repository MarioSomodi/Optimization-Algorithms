import pygad
import warnings


class KnapsackPyGAD:
    def __init__(self, values, weights, capacity, sol_per_pop=50, num_generations=100):
        self.values = values
        self.weights = weights
        self.capacity = capacity

        self.num_generations = num_generations
        self.num_parents_mating = 4
        self.init_range_low = 0
        self.init_range_high = 1
        self.sol_per_pop = sol_per_pop
        self.num_genes = len(values)

        self.ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.sol_per_pop,  # sol_per_pop is num of solutions (e.e. chromosomes) in the population
            num_genes=self.num_genes,
            init_range_low=self.init_range_low,  # lower bound for gene initialization
            init_range_high=self.init_range_high,  # upper bound for gene initialization
            gene_type=int,
            gene_space=[0, 1],  # binary
            fitness_func=self.fitness_func,
            crossover_type="uniform",
            mutation_type="random",
            mutation_percent_genes=5,
            keep_parents=0,  # no parents are kept in population
            random_seed=42,  # for reproducibility
            suppress_warnings=True
        )

    def fitness_func(self, gad, solution, solution_idx):
        solution = np.array(solution)

        total_weight = np.sum(self.weights * solution)
        total_value = np.sum(self.values * solution)

        if total_weight > self.capacity:
            return 0

        return total_value

    def run(self):
        self.ga.run()

        solution, solution_fitness, _ = self.ga.best_solution()

        print(f"Number of generations: {self.num_generations}")
        print(f"Population size: {self.sol_per_pop}")
        print(f"Number of genes: {self.num_genes}")
        print(f"Best solution: {solution}")
        print(f"Best solution fitness: {solution_fitness}")
        print(f"Selected items: {np.sum(solution)}/{len(solution)}")
        print(f"Weight constraint check: {np.sum(self.weights * solution)}/{self.capacity}")
        warnings.filterwarnings(
            "ignore",
            message="No artists with labels found to put in legend"
        )
        self.ga.plot_fitness()
        return solution, solution_fitness

from pygad import pygad
import numpy as np

from objective_functions.rastrigin import RastriginObjective


class RastriginPyGAD:
    def __init__(self, num_generations=200, sol_per_pop=100, dimensions=10):
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.dimensions = dimensions

        self.ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=10,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.dimensions,
            init_range_low=-5.12,
            init_range_high=5.12,
            gene_type=float,
            fitness_func=self.fitness_function,
            parent_selection_type="tournament",
            crossover_type="uniform",
            mutation_type="adaptive",
            mutation_percent_genes=[20, 5],  # must be tuple id adaptive
            keep_parents=4,
            K_tournament=7,
            random_mutation_min_val=-1.0,
            random_mutation_max_val=1.0,
            random_seed=42,  # for reproducibility
            suppress_warnings=True,
        )

    def fitness_function(self, gad, solution, solution_idx):
        obj = RastriginObjective(dim=self.dimensions)
        value = obj.evaluate(solution)
        return -value

    def run(self):
        self.ga.run()

        solution, solution_fitness, _ = self.ga.best_solution()

        print(f"Number of generations: {self.num_generations}")
        print(f"Population size: {self.sol_per_pop}")
        print(f"Number of genes or dimensions: {self.dimensions}")
        print(f"Best solution: {solution}")
        print(f"Best solution fitness: {-solution_fitness}")
        print(f"Distance from global optimum: {np.sqrt(np.sum(solution**2))}")

        self.ga.plot_fitness()

        return solution, -solution_fitness