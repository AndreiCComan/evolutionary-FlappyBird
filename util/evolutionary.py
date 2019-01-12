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
import pickle

import FlappyBirdClone.flappy_screen as flappy_screen
import FlappyBirdClone.flappy_no_screen as flappy_no_screen

import neat

"""
Base class for the evolutionary methods which will be used in order
to evolve flappy.
"""
class EvolutionaryModel:

    def __init__(self, args):
        self.EA = args.EA
        self.CR = args.CR
        self.F = args.F
        self.MU = args.MU
        self.NGEN = args.NGEN
        self.DEVICE = args.DEVICE
        self.MODE_AGENT = args.MODE_AGENT
        self.MODE_LEARN = args.MODE_LEARN
        self.NCPU = args.NCPU
        self.MODE_NO_SCREEN = args.MODE_NO_SCREEN

    def create_model(self, device):
        pass

    def generate_individual(self, individual, model):
        pass

    def evaluate_individual(self, model, individual):
        pass

    def evolve(self):
        pass

    def save(self):
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

        if not self.MODE_NO_SCREEN:
            score = flappy_screen.play(self.MODE_AGENT, self.MODE_LEARN, self.model, self.EA)
        else:
            score = flappy_no_screen.play(model, self.EA)

        return score

    def differential_evolution(self, agent, population):
        a, b, c = self.toolbox.select(population)
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
            return y
        else:
            return agent

    def evolve(self):

        assert self.MODE_AGENT

        for _ in tqdm(range(self.NGEN), total=self.NGEN):
            agents = [(agent, self.pop) for agent in self.pop]
            with multiprocessing.Pool(processes=self.NCPU) as pool:
                self.pop = pool.starmap(self.differential_evolution, agents)
            self.hof.update(self.pop)

        best_individual = self.hof[0]
        for parameter, numpy_array in zip(self.model.parameters(), best_individual):
            parameter.data = torch.from_numpy(numpy_array)

    def save(self):
        torch.save(self.model.state_dict(), "model.pt")


"""
EA which uses NEAT to evolve an entire Neural-Network
"""
class NEATModel(EvolutionaryModel):
    
    def __init__(self, args, config_path):
        EvolutionaryModel.__init__(self, args)

        self.config = config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    def create_network(self, genome, config):
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def evaluate_genome(self, genome, config):
        self.model = self.create_network(genome, config)

        if not self.MODE_NO_SCREEN:
            score = flappy_screen.play(self.MODE_AGENT, self.MODE_LEARN, self.model, self.EA)
        else:
            score = flappy_no_screen.play(self.model, self.EA)

        return score[0]

    def evolve(self):

        pop = neat.Population(self.config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))

        pe = neat.ParallelEvaluator(self.NCPU, self.evaluate_genome)
        self.winner = pop.run(pe.evaluate)

        return self.winner

    def save(self):
        with open('winner-feedforward', 'wb') as f:
            pickle.dump(self.winner, f)
