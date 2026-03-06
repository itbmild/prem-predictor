""" Takes processed data and performs splits on data, saving to specified directory """
from processing.loader import Loader

class SplitProvider:
    def __init__(self, config):
        self.config = config
        self.loader = Loader()
        self.data_path = config.processed
        print(self.data_path)

    def get_splits(self):
        """"""
        df = self.loader.load(self.data_path)


    