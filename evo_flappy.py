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
    parser.add_argument("--EA", type = str, default = "simple", help = "Type of Evolutionary algorithm which will be used")
    parser.add_argument("--CR", type = float, default = 0.25, help = "Crossover probability")
    parser.add_argument("--F", type = float, default = 1, help = "Differential weight")
    parser.add_argument("--MU", type = int, default = 300, help = "Population size")
    parser.add_argument("--NGEN", type = int, default = 200, help = "Number of generations")
    parser.add_argument("--DEVICE", type = str, default = "cpu", help = "Device on which to rung the PyTorch model")
    parser.add_argument("--MODE_AGENT", default = False, action = "store_true", help = "Activate agent mode")
    parser.add_argument("--MODE_LEARN", default = False, action = "store_true", help = "Activate agent learn mode")
    parser.add_argument("--MODE_NO_SCREEN", default=False, action="store_true", help="Disable screen")
    parser.add_argument("--NCPU", type = int, default = 1, help="Number of CPUs")
    parser.add_argument("--PLAY_BEST", default= False, action="store_true", help="Run the game with the best individual")

    args = parser.parse_args()

    MODE_AGENT = args.MODE_AGENT
    MODE_LEARN = args.MODE_LEARN
    EA = args.EA

    if not MODE_AGENT:

        flappy_screen.play(mode_agent = MODE_AGENT)

    elif MODE_AGENT and not MODE_LEARN:

        if (EA!="NEAT"):
            agent = evolutionary.TorchModel(args)
            agent.model.load_state_dict(torch.load("model.pt"))
            model = agent.model.double()
        else:
            with open('model-neat.pt', 'rb') as f:
                model = pickle.load(f)

        flappy_screen.play(mode_agent = MODE_AGENT, model = model, ea_type=EA)

    else:

        # Generate the model based on the type of EA
        if (EA == "NEAT"):
            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, 'config-feedforward-neat.conf')
            agent = evolutionary.NEATModel(args, config_path)
        else:
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
