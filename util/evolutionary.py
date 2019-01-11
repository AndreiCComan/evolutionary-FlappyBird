from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools

import torch
import torch.nn as nn
import numpy as np

import logging
logging.getLogger().setLevel(logging.INFO)

import random
import multiprocessing

from FlappyBirdClone.flappy import play_with_screen


"""
Base class for the evolutionary methods which will be used in order
to evolve flappy.
"""
class EvolutionaryModel:

    def __init__(self, args):
        self.CR = args.CR
        self.F = args.F
        self.MU = args.MU
        self.NGEN = args.NGEN
        self.DEVICE = args.DEVICE
        self.MODE_AGENT = args.MODE_AGENT
        self.MODE_LEARN = args.MODE_LEARN
        self.NCPU = args.NCPU

    def create_model(self, device):
        pass

    def generate_individual(self, individual, model):
        pass

    def evaluate_individual(self, model, individual):
        pass

    def evolve(self):
        pass

"""
Simple EA which evolves the weights of a PyTorch Feed-Foward Neural Network.
"""
class TorchModel(EvolutionaryModel):

    def __init__(self, args):
        EvolutionaryModel.__init__(self, args)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.model = self.create_model(self.DEVICE)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.generate_individual, creator.Individual, self.model)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selRandom, k=3)
        self.toolbox.register("evaluate", self.evaluate_individual, self.model)

        self.pop = self.toolbox.population(n=self.MU)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

    def create_model(self, device):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.LogSoftmax(dim=0))
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def generate_individual(self, individual, model):
        weights = []
        for parameter in model.parameters():
            if len(parameter.size()) == 1:
                parameter_dim = parameter.size()[0]
                weights.append(np.random.rand(parameter_dim) * np.sqrt(1 / (parameter_dim)))
            else:
                parameter_dim_0, parameter_dim_1 = parameter.size()
                weights.append(
                    np.random.rand(parameter_dim_0, parameter_dim_1) * np.sqrt(1 / (parameter_dim_0 + parameter_dim_1)))
        return individual(np.array(weights))

    def evaluate_individual(self, model, individual):
        for parameter, numpy_array in zip(model.parameters(), individual):
            parameter.data = torch.from_numpy(numpy_array)

        return play_with_screen(self.MODE_AGENT, self.MODE_LEARN, model=model)

    def differential_evolution(self, index_agent, agent):
        a, b, c = self.toolbox.select(self.pop)
        y = self.toolbox.clone(agent)
        index = random.randrange(len(agent))

        for i, value in enumerate(agent):
            if i == index or random.random() < self.CR:
                y[i] = a[i] + self.F * (b[i] - c[i])

        #for layer_index, layer_weights in enumerate(y):
        #    index = random.randrange(layer_weights.shape[0])
        #    for i, value in enumerate(layer_weights):
        #        if i == index or random.random() < self.CR:
        #            y[layer_index][i] = a[layer_index][i] + self.F * (b[layer_index][i] - c[layer_index][i])

        y.fitness.values = self.toolbox.evaluate(y)
        if y.fitness > agent.fitness:
            self.pop[index_agent] = y
        return

    def evolve(self):

        assert self.MODE_AGENT

        for _ in tqdm(range(1, self.NGEN), total=self.NGEN):
            agents = [(agent_index, agent) for agent_index, agent in enumerate(self.pop)]
            with multiprocessing.Pool(processes=self.NCPU) as pool:
                pool.starmap(self.differential_evolution, agents)

            self.hof.update(self.pop)

        best_individual = self.hof[0]
        for parameter, numpy_array in zip(self.model.parameters(), best_individual):
            parameter.data = torch.from_numpy(numpy_array)


"""
EA which uses NEAT to evolve an entire Neural-Network
"""
class NEATModel(EvolutionaryModel):
    
    def __init__(self, args):
        EvolutionaryModel.__init__(self, args)
