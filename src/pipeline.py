""" 
    Data pipeline
    Processes raw data and saves to files with defined loading, cleaning, feature and writer classes
"""
from processing.loader import Loader
from processing.writer import Writer
from processing.features import FeatureTransformer
from processing.transform import DataTransformer
import pandas as pd

COLS_TO_KEEP = [
                        "Date", "HomeTeam", "AwayTeam", 
                        "FTHG", "FTAG", "HS", "AS",
                        "HST", "AST", "FTR"
        ]

class DataPipeline:
    def __init__(self, loader: Loader, transformer: DataTransformer, writer: Writer):
        self.loader = loader
        self.transformer = transformer
        self.writer = writer

    def run(self):
        """ Runs full data pipeline """
        # load match data from files
        files = self.loader.get_files()
        seasons = self.loader.load_batch(files)

        standings = self.transformer.batch(seasons, self.transformer.get_standings)
        seasons = self.transformer.batch(seasons, lambda s: self.transformer.clean(s, COLS_TO_KEEP))

        # prepare per_team stats for aggregations
        per_team = self.transformer.batch(seasons, self.transformer.build_per_team)
        per_team = self.transformer.batch(per_team, lambda s: self.transformer.add_form(s, 5))

        # add features for each season
        seasons = self.transformer.add_features(seasons, per_team, standings)    

if __name__ == "__main__":
    transformer = DataTransformer()
    features = FeatureTransformer()
    loader = Loader('./data/raw')
    writer = Writer()

    dp = DataPipeline(loader, transformer, writer)
    dp.run()