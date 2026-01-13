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
        
        standings = self.transformer.get_batch_standings(seasons)
        seasons = self.transformer.batch_clean_seasons(files)
        per_team = self.transformer.batch_build_per_team(seasons)

        per_team = self.transformer.build_per_team_matches(seasons[0])

        print(per_team.sort_values(["Team", "Date"], ascending=[True, True])[0:40])

if __name__ == "__main__":
    transformer = DataTransformer()
    features = FeatureTransformer()
    loader = Loader('./data/raw')
    writer = Writer()

    dp = DataPipeline(loader, transformer, writer)
    dp.run()