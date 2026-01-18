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
STANDINGS_DIR = './data/processed/standings_yearly'

PROCESSED_PREFIX = 'prem-data'
STANDINGS_PREFIX = 'standings'

RAW_DATA_DIR = './data/raw'
PROCESSED_DATA_DIR = './data/processed/processed_yearly'
STARTING_YEAR = 2015
PROCESSED_FULL_DATA_PATH = './data/processed/processed_full'
FULL_MATCH_DATA_NAME = 'prem-data(2015-2025).csv'

TRAIN_DATA_PATH = "./data/processed/training"
TRAIN_DATA_FILENAME = "train_set.csv"

VAL_DATA_PATH = "./data/processed/validation"
VAL_DATA_FILENAME = "val_set.csv"

TEST_DATA_PATH = "./data/processed/test"
TEST_DATA_FILENAME = "test_set.csv"

class DataPipeline:
    def __init__(self, loader: Loader, transformer: DataTransformer, writer: Writer):
        self.loader = loader
        self.transformer = transformer
        self.writer = writer

    def run(self):
        """ Runs full data pipeline """
        # load match data from files
        files = self.loader.get_files(RAW_DATA_DIR)
        seasons = self.loader.load_batch(files)

        standings = self.transformer.batch(seasons, self.transformer.get_standings)
        self.writer.batch_save_to_dir(standings, STANDINGS_DIR, STARTING_YEAR, STANDINGS_PREFIX)
        seasons = self.transformer.batch(seasons, lambda s: self.transformer.clean(s, COLS_TO_KEEP))

        # prepare per_team stats for aggregations
        per_team = self.transformer.batch(seasons, self.transformer.build_per_team)
        per_team = self.transformer.batch(per_team, lambda s: self.transformer.add_form(s, 5))

        # add features for each season
        seasons = self.transformer.add_features(seasons, per_team, standings)
        # print(seasons)

        # save seasons to file
        self.writer.batch_save_to_dir(seasons, PROCESSED_DATA_DIR, STARTING_YEAR, PROCESSED_PREFIX) 

        seasons_concat = self.transformer.concat_dfs(seasons)

        self.writer.save_to_dir(seasons_concat, PROCESSED_FULL_DATA_PATH, FULL_MATCH_DATA_NAME)

        train_df, val_df, test_df = self.transformer.get_splits(seasons, 6, 1, 2)
        self.writer.save_to_dir(train_df, TRAIN_DATA_PATH, TRAIN_DATA_FILENAME)
        self.writer.save_to_dir(val_df, VAL_DATA_PATH, VAL_DATA_FILENAME)
        self.writer.save_to_dir(test_df, TEST_DATA_PATH, TEST_DATA_FILENAME)


if __name__ == "__main__":
    transformer = DataTransformer()
    loader = Loader()
    writer = Writer()

    dp = DataPipeline(loader, transformer, writer)
    dp.run()