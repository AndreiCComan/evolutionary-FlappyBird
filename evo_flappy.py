from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools
from util.trivial import log_flappy_bird
log_flappy_bird()

from FlappyBirdClone.flappy import play_with_screen

import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import multiprocessing
import logging
import copy
logging.getLogger().setLevel(logging.INFO)

creator.create("FitnessMax", base.Fitness, weights = (1.0, ))
creator.create("Individual", np.ndarray, fitness = creator.FitnessMax)

# ----------------------------------------------------------------------------------------------------------------------


def create_model(device):
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.LogSoftmax(dim = 0))
    for parameter in model.parameters():
        parameter.requires_grad = False
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


def evaluate_individual(model, args, individual):
    for parameter, numpy_array in zip(model.parameters(), individual):
        parameter.data = torch.from_numpy(numpy_array)

    return play_with_screen(args.MODE_AGENT, args.MODE_LEARN, model = model)


def main():

    parser = argparse.ArgumentParser("evo_flappy.py")
    parser.add_argument("--CR", type = float, default = 0.25, help = "Crossover probability")
    parser.add_argument("--F", type = float, default = 1, help = "Differential weight")
    parser.add_argument("--MU", type = int, default = 300, help = "Population size")
    parser.add_argument("--NGEN", type = int, default = 200, help = "Number of generations")
    parser.add_argument("--DEVICE", type = str, default = "cpu", help = "Device on which to rung the PyTorch model")
    parser.add_argument("--MODE_AGENT", default = False, action = "store_true", help = "Activate agent mode")
    parser.add_argument("--MODE_LEARN", default = False, action = "store_true", help = "Activate agent learn mode")

    args = parser.parse_args()

    MODE_AGENT = args.MODE_AGENT
    MODE_LEARN = args.MODE_LEARN

    if not MODE_AGENT:

        play_with_screen(mode_agent = MODE_AGENT)

    elif MODE_AGENT and not MODE_LEARN:

        model = create_model(args.DEVICE)
        model.load_state_dict(torch.load("model.pt"))
        model = model.double()
        play_with_screen(mode_agent = MODE_AGENT, model = model)

    else:

        assert args.MODE_AGENT

        CR = args.CR
        F = args.F
        MU = args.MU
        NGEN = args.NGEN

        DEVICE = args.DEVICE

        logging.info("Creating Neural Network model")
        model = create_model(DEVICE)

        pool = multiprocessing.Pool()

        toolbox = base.Toolbox()
        toolbox.register("map", pool.map)
        toolbox.register("individual", generate_individual, creator.Individual, model)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k = 3)
        toolbox.register("evaluate", evaluate_individual, model, args)

        pop = toolbox.population(n = MU)
        hof = tools.HallOfFame(1, similar = np.array_equal)

        for _ in tqdm(range(1, NGEN), total = NGEN):
            for index_agent, agent in tqdm(enumerate(pop), total = len(pop)):
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

        best_individual = hof[0]
        for parameter, numpy_array in zip(model.parameters(), best_individual):
            parameter.data = torch.from_numpy(numpy_array)

        torch.save(model.state_dict(), "model.pt")
        play_with_screen(mode_agent = MODE_AGENT, model = model)


if __name__ == "__main__":
    main()