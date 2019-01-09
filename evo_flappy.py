import random
import numpy
import argparse

import logging
logging.getLogger().setLevel(logging.INFO)

from trivial import log_flappy_bird
log_flappy_bird()

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

# ----------------------------------------------------------------------------------------------------------------------


def generate_individual(individual):
    # TODO
    return individual(numpy.random.rand(10))


def evaluate_individual(individual):
    # TODO
    return 1,

# ----------------------------------------------------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser("evo_flappy.py")
    parser.add_argument("--CR", type = float, default = 0.25, help = "Crossover probability")
    parser.add_argument("--F", type = float, default = 1, help = "Differential weight")
    parser.add_argument("--MU", type = int, default = 300, help= "Population size")
    parser.add_argument("--NGEN", type = int, default = 200, help = "Number of generations")
    parser.add_argument("--seed", default = -1, help="set the random seed")
    args = parser.parse_args()

    if args.seed != -1:
        random.seed(args.seed)

    CR = args.CR
    F = args.F
    MU = args.MU
    NGEN = args.NGEN

    creator.create("FitnessMax", base.Fitness, weights = (1.0, ))
    creator.create("Individual", numpy.ndarray, fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom, k = 3)
    toolbox.register("evaluate", evaluate_individual)

    pop = toolbox.population(n = MU)
    hof = tools.HallOfFame(1, similar = numpy.array_equal)

    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for generation in tqdm(range(1, NGEN), total = NGEN):

        for index_agent, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(10)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[index_agent] = y
        hof.update(pop)

    logging.info("Best individual is {} {}".format(hof[0], hof[0].fitness.values[0]))


if __name__ == "__main__":
    main()