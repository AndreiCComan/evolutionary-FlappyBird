import configparser

class ExperimentParser:

    def __init__(self, args, config_path):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
        self.args = args

    def parse(self):
        self.args.algorithms = self.getlist(self.parser["Default"]["algorithms"])
        self.args.difficulties = self.getlist(self.parser["Default"]["difficulties"])
        self.args.population_sizes = self.getlist(self.parser["Default"]["population_sizes"])
        self.args.generations = self.getlist(self.parser["Default"]["generations"])
        return self.args


    def getlist(self, option, sep=',', chars=None):
        """Return a list from a ConfigParser option. By default,
           split on a comma and strip whitespaces."""
        return [chunk.strip(chars) for chunk in option.split(sep)]


