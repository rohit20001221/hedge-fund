import numpy as np

def expected_returns(weights, log_returns):
    return np.sum(np.mean(log_returns) * weights)

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    
    return np.sqrt(variance)

def sharp_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_returns(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)

def random_genome(num_stocks):
    return np.random.randn(num_stocks)

def init_population(population_size, genome_length):
    return [
        random_genome(genome_length)
        for _ in range(population_size)
    ]

def fitness(genome, log_returns, cov_matrix):
    weights = softmax(genome)

    return sharp_ratio(
        weights,
        log_returns,
        cov_matrix,
        0
    )

def select_parent(population, fitness_values):
    fitness_values = np.asarray(fitness_values)

    adjusted_fitness = (
        fitness_values
        - np.min(fitness_values)
        + 1e-8
    )

    total_fitness = np.sum(adjusted_fitness)
    pick = np.random.uniform(0, total_fitness)
    current = 0

    for indivudial, fitness_value in zip(population, adjusted_fitness):
        current += fitness_value
        if current >= pick:
            return indivudial

    return population[-1]

def crossover(parent_a, parent_b, crossover_rate=0.7):
    if np.random.random() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent_a))

        child_a = np.concatenate([
            parent_a[:crossover_point],
            parent_b[crossover_point:]
        ])

        child_b = np.concatenate([
            parent_b[:crossover_point],
            parent_a[crossover_point:]
        ])

        return child_a, child_b

    return parent_a, parent_b

def mutate(genome, mutation_rate=0.01):
    genome = genome.copy()

    for i in range(len(genome)):
        if np.random.random() < mutation_rate:
            genome[i] += np.random.normal(
                loc=0.0,
                scale=0.2
            )

    return genome

def genetic_algorithm(log_returns, cov_matrix, genome_length, population_size=500, num_generations=500):
    population = init_population(
        population_size,
        genome_length
    )

    fitness_history = []
    for generation in range(num_generations):
        fitness_values = np.array([
            fitness(genome, log_returns, cov_matrix)
            for genome in population
        ])

        new_population = []
        for _ in range(population_size // 2):
            parent_a = select_parent(population, fitness_values)
            parent_b = select_parent(population, fitness_values)

            child_a, child_b = crossover(parent_a, parent_b)

            child_a = mutate(child_a)
            child_b = mutate(child_b)

            new_population.extend([child_a, child_b])

        population = new_population[:population_size]

        best_idx = np.argmax(fitness_values)
        best_fitness = fitness_values[best_idx]

        fitness_history.append(best_fitness)

        print(
            f"Generation {generation + 1} / {num_generations}"
            f"| Best Sharpe = {best_fitness:.4f}"
        )

    # final evaluation
    fitness_values = np.array([fitness(genome, log_returns, cov_matrix) for genome in population])

    idx = np.argmax(fitness_values)

    best_genome = population[idx]
    best_fitness = fitness_values[idx]

    best_solution = softmax(best_genome)

    return {
        "weights": best_solution, 
        "best_fitness": best_fitness, 
        "fitness_history": fitness_history,
    }

