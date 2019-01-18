import configparser


class ExperimentParser:

    def __init__(self, args, config_path):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
        self.args = args

    def parse(self):

        self.args.ALGORITHMS = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["algorithms"])
        self.args.DIFFICULTIES = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["difficulties"])
        self.args.POPULATION_SIZES = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["population_sizes"])
        self.args.GENERATIONS = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["generations"])

        if self.args.CONFIGURATION_TYPE == "DifferentialEvolution":
            assert self.args.EA == "EA"
            self.args.ARCHITECTURES = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["architecture"])
            self.args.WEIGHTS_UPDATES = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["weights_update"])

        elif self.args.CONFIGURATION_TYPE == "NEAT":
            assert self.args.EA == "NEAT"
            self.args.ELITISMS = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["elitism"])
            self.args.HIDDEN_LAYER_SIZES = self.getlist(self.parser[self.args.CONFIGURATION_TYPE]["hidden_layer_size"])

        return self.args


    def getlist(self, option, sep=',', chars=None):
        """Return a list from a ConfigParser option. By default,
           split on a comma and strip whitespaces."""
        return [chunk.strip(chars) for chunk in option.split(sep)]


