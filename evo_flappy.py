from util.ExperimentParser import ExperimentParser
from util.trivial import log_flappy_bird
log_flappy_bird()

import FlappyBirdClone.flappy_screen as flappy_screen

import os
import argparse
import torch
import logging
logging.getLogger().setLevel(logging.INFO)

from util import evolutionary

import pickle

def main():

    parser = argparse.ArgumentParser("evo_flappy.py")
    parser.add_argument("--EA", type = str, default = None, help = "Type of Evolutionary algorithm which will be used")
    parser.add_argument("--CR", type = float, default = 0.25, help = "Crossover probability")
    parser.add_argument("--F", type = float, default = 1, help = "Differential weight")
    parser.add_argument("--MU", type = int, default = 300, help = "Population size")
    parser.add_argument("--NGEN", type = int, default = 200, help = "Number of generations")
    parser.add_argument("--DEVICE", type = str, default = "cpu", help = "Device on which to rung the PyTorch model")
    parser.add_argument("--DIFFICULTY", type=str, default="hard", help="Difficulty of the game.")
    parser.add_argument("--MODE_AGENT", default = False, action = "store_true", help = "Activate agent mode")
    parser.add_argument("--MODE_LEARN", default = False, action = "store_true", help = "Activate agent learn mode")
    parser.add_argument("--EXPERIMENTS", default = False, action = "store_true", help= "Execute experiments specified in the config file.")
    parser.add_argument("--MODE_NO_SCREEN", default=False, action="store_true", help="Disable screen")
    parser.add_argument("--NCPU", type = int, default = 1, help="Number of CPUs")
    parser.add_argument("--PLAY_BEST", default= False, action="store_true", help="Run the game with the best individual")
    parser.add_argument("--LOG_PERFORMANCE", default=False, action="store_true", help="Log some performance informations to files.")
    parser.add_argument("--MODEL_FILE", type=str, default="", help="Location of the pre-trained model we want to evalute.")

    args = parser.parse_args()

    MODE_AGENT = args.MODE_AGENT
    MODE_LEARN = args.MODE_LEARN
    EA = args.EA

    if not MODE_AGENT:

        flappy_screen.play(mode_agent = MODE_AGENT)

    elif MODE_AGENT and not MODE_LEARN:

        if args.MODEL_FILE != "":
            model_file = args.MODEL_FILE
        else:
            model_file = "model.pt" if EA != "NEAT" else "model-neat.pt"

        if (EA!="NEAT"):
            args.ARCHITECTURE = "shallow"
            args.WEIGHTS_UPDATE = "shallow"
            agent = evolutionary.TorchModel(args)
            agent.model.load_state_dict(torch.load(model_file))
            model = agent.model.double()
        else:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

        flappy_screen.play(mode_agent = MODE_AGENT, model = model, ea_type=EA, difficulty = args.DIFFICULTY)

    elif args.EXPERIMENTS:
        exp_parser = ExperimentParser(args, "./experiments.conf")

        args = exp_parser.parse()

        # For each possible algorithm, difficulties and generation,
        # train a model
        for algorithm in args.ALGORITHMS:
            for difficulty in args.DIFFICULTIES:
                for ngen in args.GENERATIONS:
                    for population_size in args.POPULATION_SIZES:

                        args.DIFFICULTY = difficulty
                        args.NGEN = int(ngen)
                        args.MU = int(population_size)
                        args.EA = algorithm

                        # Generate the model based on the type of EA
                        if (algorithm == "NEAT"):
                            for eli in args.ELITISMS:
                                for size_l in args.HIDDEN_LAYER_SIZES:
                                    args.ELITISM = int(eli)
                                    args.HIDDEN_LAYER_SIZE = int(size_l)
                                    logging.info(
                                        "[*] Algorithm {}; Difficulty {}; Generations {}; Individuals {}; Elitism {}; Hidden Layer Size {}"
                                            .format(args.EA,
                                                    args.DIFFICULTY,
                                                    args.NGEN,
                                                    args.MU,
                                                    args.ELITISM,
                                                    args.HIDDEN_LAYER_SIZE))
                                    local_dir = os.path.dirname(__file__)
                                    config_path = os.path.join(local_dir, 'config-feedforward-neat.conf')
                                    agent = evolutionary.NEATModel(args, config_path)
                                    agent.evolve()
                                    agent.save()
                        else:
                            for architecture in args.ARCHITECTURES:
                                for weights_update in args.WEIGHTS_UPDATES:
                                    args.ARCHITECTURE = architecture
                                    args.WEIGHTS_UPDATE = weights_update
                                    logging.info(
                                        "[*] Algorithm {}; Difficulty {}; Generations {}; Individuals {}; Architecture {}; Weights Update {}"
                                            .format(args.EA,
                                                    args.DIFFICULTY,
                                                    args.NGEN,
                                                    args.MU,
                                                    args.ARCHITECTURE,
                                                    args.WEIGHTS_UPDATE))
                                    agent = evolutionary.TorchModel(args)
                                    agent.evolve()
                                    agent.save()

    else:

        # Generate the model based on the type of EA
        if (EA == "NEAT"):
            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, 'config-feedforward-neat.conf')
            agent = evolutionary.NEATModel(args, config_path)
        else:
            args.ARCHITECTURE = "shallow"
            args.WEIGHTS_UPDATE = "shallow"
            agent = evolutionary.TorchModel(args)

        # Evolve the model
        agent.evolve()

        # Save the best model on file and run it
        agent.save()

        # Run the best
        if args.PLAY_BEST:
            flappy_screen.play(ea_type=EA, mode_agent = MODE_AGENT, model = agent.model)


if __name__ == "__main__":
    main()
