import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Parameters
GRID_SIZE = 10  # Grid size (10x10)
POPULATION_SIZE = 20
GENERATIONS = 50
MUTATION_RATE = 0.1
OUTPUT_FOLDER = "output"

# Define terrain types
TERRAIN_TYPES = {0: "Plain", 1: "River", 2: "Forest", 3: "Mountain"}
TERRAIN_COLORS = {0: "#a1d99b", 1: "#6baed6", 2: "#31a354", 3: "#756bb1"}

# Fitness function weights
WEIGHTS = {
    "connectivity": 0.4,
    "diversity": 0.3,
    "natural_flow": 0.2,
    "key_features": 0.1
}

def generate_random_landscape():
    """Generate a random landscape."""
    return np.random.choice(list(TERRAIN_TYPES.keys()), size=(GRID_SIZE, GRID_SIZE))

def calculate_fitness(landscape):
    """Evaluate the fitness of a landscape."""
    # Dummy fitness calculations (improve based on specific rules)
    connectivity = np.sum(landscape == 1) / (GRID_SIZE ** 2)  # Proportion of river
    diversity = len(np.unique(landscape)) / len(TERRAIN_TYPES)  # Diversity of terrains
    natural_flow = 1 if np.any(landscape[0, :] == 1) and np.any(landscape[-1, :] == 1) else 0  # River flows from top to bottom
    key_features = 1 if np.any(landscape == 2) and np.any(landscape == 3) else 0  # Forest and Mountain present

    fitness = (
        WEIGHTS["connectivity"] * connectivity +
        WEIGHTS["diversity"] * diversity +
        WEIGHTS["natural_flow"] * natural_flow +
        WEIGHTS["key_features"] * key_features
    )
    return fitness

def mutate(landscape):
    """Mutate a landscape."""
    for _ in range(int(GRID_SIZE * GRID_SIZE * MUTATION_RATE)):
        x, y = np.random.randint(0, GRID_SIZE, size=2)
        landscape[x, y] = random.choice(list(TERRAIN_TYPES.keys()))
    return landscape

def crossover(parent1, parent2):
    """Perform crossover between two landscapes."""
    point = random.randint(1, GRID_SIZE - 1)
    child = np.vstack((parent1[:point, :], parent2[point:, :]))
    return child

def save_landscape(landscape, generation, idx):
    """Save a landscape as an image file."""
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.matplotlib.colors.ListedColormap([TERRAIN_COLORS[t] for t in TERRAIN_TYPES])
    ax.imshow(landscape, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f"Generation {generation} - Landscape {idx}")
    plt.savefig(f"{OUTPUT_FOLDER}/landscape_{generation}_{idx}.png")
    plt.close()

def main():
    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Initialize population
    population = [generate_random_landscape() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = [calculate_fitness(landscape) for landscape in population]

        # Select top individuals (elitism)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices[:POPULATION_SIZE // 2]]

        # Generate offspring
        offspring = []
        for _ in range(POPULATION_SIZE - len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            offspring.append(child)

        population.extend(offspring)

        # Save top landscapes
        for idx, top_landscape in enumerate(population[:10]):
            save_landscape(top_landscape, generation, idx)

        print(f"Generation {generation} completed. Best fitness: {fitness_scores[sorted_indices[0]]:.2f}")

if __name__ == "__main__":
    main()
