import random
import numpy as np
import argparse
import torch
import time
import torch.nn as nn

import logging
logging.getLogger().setLevel(logging.INFO)

from trivial import log_flappy_bird
log_flappy_bird()

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

# ----------------------------------------------------------------------------------------------------------------------


def create_model(device):
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.LogSoftmax(dim = 0))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device)
    return model


def generate_individual(individual, model):
    weights = []
    for parameter in model.parameters():
        if len(parameter.size()) == 1:
            parameter_dim = parameter.size()[0]
            weights.append(np.random.rand(parameter_dim) * np.sqrt(1 / (parameter_dim)))
        else:
            parameter_dim_0, parameter_dim_1 = parameter.size()
            weights.append(np.random.rand(parameter_dim_0, parameter_dim_1) * np.sqrt(1 / (parameter_dim_0 + parameter_dim_1)))
    return individual(np.array(weights))


def evaluate_individual(model, individual):
    for parameter, numpy_array in zip(model.parameters(), individual):
        parameter.data = torch.from_numpy(numpy_array)
    output_tensor = model(torch.tensor([0.1, 0.2, 0.3], dtype = torch.double))
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

    GPU_ID = 0
    DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
    if str(DEVICE) == "cpu":
        logging.warning("Running model on CPU")
    else:
        logging.info("Running model on GPU {}".format(GPU_ID))

    logging.info("Creating Neural Network model")
    model = create_model(DEVICE)

    creator.create("FitnessMax", base.Fitness, weights = (1.0, ))
    creator.create("Individual", np.ndarray, fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, creator.Individual, model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom, k = 3)
    toolbox.register("evaluate", evaluate_individual, model)

    pop = toolbox.population(n = MU)

    hof = tools.HallOfFame(1, similar = np.array_equal)

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
                    return
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[index_agent] = y
        hof.update(pop)

    logging.info("Best individual is {} {}".format(hof[0], hof[0].fitness.values[0]))


if __name__ == "__main__":
    main()