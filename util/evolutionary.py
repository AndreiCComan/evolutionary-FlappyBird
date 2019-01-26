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

# Custom logger for each of the objects. It has to be
# a global variable since it is not serializable and it
# cannot be saved inside the EvolutionaryModel objects.
# By setting propagate to False we disable the standard
# stdout output.
global logger
logger = logging.getLogger("performance_logger")
logger.propagate = False

# File handler used by the loggers. It has also to be global
# because it is not serializable. It must be removed from
# the logger once the evolve phase is finished.
global file_handler
file_handler = None


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
        self.LOG_PERFORMANCE = args.LOG_PERFORMANCE
        self.DIFFICULTY = args.DIFFICULTY

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

    def __str__(self):
        return self.__class__.__name__

    # Remove the file handler from the logger. This will enable us
    # to run several time the experiments while changing the final
    # output file.
    def remove_handler(self):
        if self.LOG_PERFORMANCE:
            logger.removeHandler(file_handler)

"""
Simple EA which evolves the weights of a PyTorch Feed-Foward Neural Network.
"""
class TorchModel(EvolutionaryModel):

    def __init__(self, args):
        EvolutionaryModel.__init__(self, args)

        self.ARCHITECTURE = args.ARCHITECTURE
        self.WEIGHTS_UPDATE = args.WEIGHTS_UPDATE

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

        # Set up the (global) file handler to which we will save things
        global file_handler

        if self.LOG_PERFORMANCE:
            file_handler = logging.FileHandler("./experiments/DE/EA_{}_DIFFICULTY_{}_NGEN_{}_MU_{}_ARCHITECTURE_{}_WEIGHTS_UPDATE_{}_performance.csv"
                   .format(self.EA,
                           self.DIFFICULTY,
                           self.NGEN,
                           self.MU,
                           self.ARCHITECTURE,
                           self.WEIGHTS_UPDATE))
            logger.addHandler(file_handler)

    def create_model(self, device):
        model = None

        assert self.ARCHITECTURE == "shallow" or self.ARCHITECTURE == "deep" or self.ARCHITECTURE == "wide" or self.ARCHITECTURE == "wider"

        if self.ARCHITECTURE == "shallow":
            model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.LogSoftmax(dim=0))
        elif self.ARCHITECTURE == "deep":
            model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU(), nn.Linear(4,2), nn.LogSoftmax(dim=0))
        elif self.ARCHITECTURE == "wide":
            model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2), nn.LogSoftmax(dim=0))
        elif self.ARCHITECTURE == "wider":
            model = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 2), nn.LogSoftmax(dim=0))

        assert model is not None

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
            score = flappy_screen.play(self.MODE_AGENT, self.MODE_LEARN, self.model, self.EA, self.DIFFICULTY)
        else:
            score = flappy_no_screen.play(model, self.EA, self.DIFFICULTY)

        return score

    def differential_evolution(self, agent, population):
        a, b, c = self.toolbox.select(population)
        y = self.toolbox.clone(agent)
        index = random.randrange(len(agent))

        assert self.WEIGHTS_UPDATE == "shallow" or self.WEIGHTS_UPDATE == "normal"

        if self.WEIGHTS_UPDATE == "shallow":
            for i, value in enumerate(agent):
                if i == index or random.random() < self.CR:
                    y[i] = a[i] + self.F * (b[i] - c[i])
        else: # self.WEIGHTS_UPDATE == "normal"
            for layer_index, layer_weights in enumerate(y):
                index = random.randrange(layer_weights.shape[0])
                for i, value in enumerate(layer_weights):
                    if i == index or random.random() < self.CR:
                        y[layer_index][i] = a[layer_index][i] + self.F * (b[layer_index][i] - c[layer_index][i])

        y.fitness.values = self.toolbox.evaluate(y)
        if y.fitness > agent.fitness:
            return y
        else:
            return agent

    def evolve(self):

        assert self.MODE_AGENT

        logger.info("generation,mean_fitness")

        for generation in tqdm(range(self.NGEN), total=self.NGEN):
            agents = [(agent, self.pop) for agent in self.pop]
            with multiprocessing.Pool(processes=self.NCPU) as pool:
                self.pop = pool.starmap(self.differential_evolution, agents)
            self.hof.update(self.pop)

            population_fitness = 0
            for individual in self.pop:
                population_fitness += individual.fitness.values[0]

            logger.info("{},{}".format(generation, population_fitness / len(self.pop)))


        best_individual = self.hof[0]
        for parameter, numpy_array in zip(self.model.parameters(), best_individual):
            parameter.data = torch.from_numpy(numpy_array)

        logger.removeHandler(file_handler)

    def save(self):
        torch.save(self.model.state_dict(), "./experiments/DE/EA_{}_DIFFICULTY_{}_NGEN_{}_MU_{}_ARCHITECTURE_{}_WEIGHTS_UPDATE_{}_model.pt"
                   .format(self.EA,
                           self.DIFFICULTY,
                           self.NGEN,
                           self.MU,
                           self.ARCHITECTURE,
                           self.WEIGHTS_UPDATE))


"""
EA which uses NEAT to evolve an entire Neural-Network
"""
class NEATModel(EvolutionaryModel):

    def __init__(self, args, config_path):
        EvolutionaryModel.__init__(self, args)

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        self.model = ''
        self.winner = ''

        self.ELITISM = int(args.ELITISM)
        self.HIDDEN_LAYER_SIZE = int(args.HIDDEN_LAYER_SIZE)

        self.config.pop_size = args.MU
        self.config.reproduction_config.elitism = self.ELITISM
        self.config.genome_config.num_hidden = self.HIDDEN_LAYER_SIZE

        # Set up the (global) file handler to which we will save things
        global file_handler

        if self.LOG_PERFORMANCE:
            file_handler = logging.FileHandler(
                "./experiments/NEAT/EA_{}_DIFFICULTY_{}_NGEN_{}_MU_{}_ELITISM_{}_HIDDEN_SIZE_{}_performance.csv"
                .format(self.EA,
                        self.DIFFICULTY,
                        self.NGEN,
                        self.MU,
                        self.ELITISM,
                        self.HIDDEN_LAYER_SIZE))
            logger.addHandler(file_handler)

    def create_network(self, genome, config):
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def evaluate_genome(self, genome, config):
        self.model = self.create_network(genome, config)

        if not self.MODE_NO_SCREEN:
            score = flappy_screen.play(self.MODE_AGENT, self.MODE_LEARN, self.model, self.EA, difficulty=self.DIFFICULTY)
        else:
            score = flappy_no_screen.play(self.model, self.EA, difficulty=self.DIFFICULTY)

        return score[0]

    def evolve(self):

        logger.info("generation,best_fitness,mean_fitness,median_fitness")

        pop = neat.Population(self.config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))

        pe = neat.ParallelEvaluator(self.NCPU, self.evaluate_genome)
        self.winner = pop.run(pe.evaluate, self.NGEN)
        self.model = neat.nn.FeedForwardNetwork.create(self.winner, self.config)

        best_genomes = stats.most_fit_genomes
        fitness_mean = stats.get_fitness_mean()
        fitness_median = stats.get_fitness_median()
        for e in range(0, self.NGEN):
            logger.info("{},{},{},{}".format(e, best_genomes[e].fitness, fitness_mean[e], fitness_median[e]))

        logger.removeHandler(file_handler)

    def save(self):
        with open("./experiments/NEAT/EA_{}_DIFFICULTY_{}_NGEN_{}_MU_{}_ELITISM_{}_HIDDEN_SIZE_{}_model.pt"
                .format(self.EA,
                        self.DIFFICULTY,
                        self.NGEN,
                        self.MU,
                        self.ELITISM,
                        self.HIDDEN_LAYER_SIZE), 'wb') as f:
            pickle.dump(self.model, f)
        with open("./experiments/NEAT/EA_{}_DIFFICULTY_{}_NGEN_{}_MU_{}_ELITISM_{}_HIDDEN_SIZE_{}_network.pt"
                          .format(self.EA,
                                  self.DIFFICULTY,
                                  self.NGEN,
                                  self.MU,
                                  self.ELITISM,
                                  self.HIDDEN_LAYER_SIZE), 'wb') as f:
            pickle.dump(self.winner, f)
