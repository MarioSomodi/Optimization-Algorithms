import numpy as np
import pandas as pd
from algorithms.genetic_optimization import GeneticAlgorithm
from objective_functions.rastrigin import RastriginObjective

# --- Hyperparameter grids ---
pop_sizes = [50, 100, 200]
crossover_methods = ['single_point', 'two_point', 'uniform']
mutation_rates = [0.01, 0.05, 0.1]
selection_methods = ['tournament', 'roulette']

# Fixed GA settings for Rastrigin (10D)
dim = 10
max_generations = 200
crossover_rate = 0.9
mutation_scale = 0.1

# Threshold for poor convergence analysis
DEFAULT_THRESHOLD = 0.1


def evaluate_ga(pop_size, crossover_method, mutation_rate, selection_method):
    """
    Run a single GA instance on the Rastrigin function and return
    (best_fitness, generation_of_best).
    """
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
        tournament_size=3,
        crossover_method=crossover_method,
        mutation_method='gaussian',
        mutation_scale=mutation_scale
    )

    best_fit = -np.inf
    gen_of_best = 0

    for generation in range(1, max_generations + 1):
        fits = ga.evaluate_population()
        idx = np.argmax(fits)
        if fits[idx] > best_fit:
            best_fit = fits[idx]
            gen_of_best = generation
        # advance population
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = ga.select_parent(fits)
            p2 = ga.select_parent(fits)
            c1, c2 = ga.crossover(p1, p2)
            new_pop.append(ga.mutate(c1))
            if len(new_pop) < pop_size:
                new_pop.append(ga.mutate(c2))
        ga.population = np.array(new_pop)

    return best_fit, gen_of_best


def run_hyperparameter_experiments(
    pop_sizes=pop_sizes,
    crossover_methods=crossover_methods,
    mutation_rates=mutation_rates,
    selection_methods=selection_methods,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Runs GA experiments over all combinations of hyperparameters,
    returns a DataFrame of results and a DataFrame of poor-convergence cases.

    Args:
        pop_sizes: list of population sizes
        crossover_methods: list of crossover types
        mutation_rates: list of mutation rates
        selection_methods: list of selection methods
        threshold: fitness threshold to flag poor convergence
        save_csv: whether to save results to CSV
        csv_path: path for CSV output

    Returns:
        df_results (pd.DataFrame), df_poor (pd.DataFrame)
    """
    records = []
    for pop in pop_sizes:
        for cross in crossover_methods:
            for mut in mutation_rates:
                for sel in selection_methods:
                    best_fit, gen_best = evaluate_ga(pop, cross, mut, sel)
                    records.append({
                        'pop_size': pop,
                        'crossover': cross,
                        'mutation_rate': mut,
                        'selection': sel,
                        'best_fitness': best_fit,
                        'gen_of_best': gen_best
                    })

    df_results = pd.DataFrame(records)

    df_poor = df_results[df_results['best_fitness'] < threshold]
    return df_results, df_poor
