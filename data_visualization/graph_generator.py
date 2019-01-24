import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("graph_generator.py")
    parser.add_argument("--data_path", type=str, default="../experiments/", help="Location of the CSV files.")
    parser.add_argument("--algorithms", nargs='+', default=["DE"], help="Filter results based on the algorithms.")
    parser.add_argument("--metric", nargs='+', default=["mean_fitness"], help="Choose metric which will be plotted.")
    parser.add_argument("--difficulty", nargs='+', default=["hard"], help="Filter results based on the difficulty.")
    parser.add_argument("--generations", type=int, default="100", help="Filter based on the number of generations.")
    parser.add_argument("--save", default=False, action="store_true", help="Save the plot to a file.")
    parser.add_argument("--baseline", default=-1, type=int, help="Add a baseline to the plot.")

    args = parser.parse_args()

    data = args.data_path
    difficulty= args.difficulty
    algorithm = args.algorithms
    metric = args.metric
    gen = args.generations
    baseline = args.baseline

    plt.figure(figsize=(10, 5))

    for a in algorithm:
        for f in os.listdir(data+a):
            if f.endswith(".csv"):

                df = pd.read_csv(data+a+"/"+f)

                # Split the filename in order to get all informations
                # 1: ea type
                # 3: game difficulty
                # 5: num generations
                # 7: population size
                # 9: elitism / architecture type (it depends on NEAT or DE)
                # 12: hidden layer size / weights update (it depends on NEAT or DE)
                components = f.split("_")

                # Plot each of the metric, checking if the file we are looking
                # for has specific characteristics
                for m in metric:
                    if (int(components[5]) == gen and components[3] in difficulty and components[1]==a):

                        label_name="{}_{}_{}_{}".format(a, components[3], components[9], components[12])
                        plt.plot(range(0, gen), df[metric], label=label_name)

    # Add the baseline
    if baseline != -1:
        plt.plot(range(0, 100), [baseline for _ in range(0,100)], label="Baseline")

    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Performance Evolutionary Flappy Bird")
    plt.legend()


    if (args.save):
        plt.savefig("plot.png", dpi=200)
    else:
        plt.show()
