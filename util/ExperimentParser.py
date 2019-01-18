import configparser


class ExperimentParser:

    def __init__(self, args, config_path):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
        self.args = args

    def parse(self):

        self.args.ALGORITHMS = self.getlist(self.parser["Default"]["algorithms"])
        self.args.DIFFICULTIES = self.getlist(self.parser["Default"]["difficulties"])
        self.args.POPULATION_SIZES = self.getlist(self.parser["Default"]["population_sizes"])
        self.args.GENERATIONS = self.getlist(self.parser["Default"]["generations"])

        self.args.ARCHITECTURES = self.getlist(self.parser["DifferentialEvolution"]["architecture"])
        self.args.WEIGHTS_UPDATES = self.getlist(self.parser["DifferentialEvolution"]["weights_update"])

        self.args.ELITISMS = self.getlist(self.parser["NEAT"]["elitism"])
        self.args.HIDDEN_LAYER_SIZES = self.getlist(self.parser["NEAT"]["hidden_layer_size"])

        return self.args


    def getlist(self, option, sep=',', chars=None):
        """Return a list from a ConfigParser option. By default,
           split on a comma and strip whitespaces."""
        return [chunk.strip(chars) for chunk in option.split(sep)]


