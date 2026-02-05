""" 
    Data pipeline
    Processes raw data and saves to files with defined loading, cleaning, feature and writer classes
"""
from processing.loader import Loader
from processing.writer import Writer
from processing.transform import DataTransformer
from processing.features import RollingWindowFeatures, HeadToHeadFeatures, PrevSeasonFeatures
from utils import PREV_SEASON_COLS, BASELINE_POS, RENAME_DICT, WINDOW_SIZE
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
STARTING_YEAR = 2008
PROCESSED_FULL_DATA_PATH = './data/processed/processed_full'
FULL_MATCH_DATA_NAME = 'prem-data(2015-2025).csv'
TRAIN_DATA_PATH = "./data/processed/training"
TRAIN_DATA_FILENAME = "train_set.csv"
VAL_DATA_PATH = "./data/processed/validation"
VAL_DATA_FILENAME = "val_set.csv"
TEST_DATA_PATH = "./data/processed/test"
TEST_DATA_FILENAME = "test_set.csv"
PER_MATCH_PATH = "./data/processed/per_team"

class DataPipeline:
    def __init__(self, loader: Loader, transformer: DataTransformer, writer: Writer):
        self.loader = loader
        self.transformer = transformer
        self.writer = writer

    def _get_raw_data(self) -> pd.DataFrame:
        """
        Loads raw match data from file and returns pandas dataframe containing the data
        """

        files = self.loader.get_files(RAW_DATA_DIR)

        print("\n--- [DEBUG] OS File Loading Order ---")
        for i, f in enumerate(files):
            print(f"Index {i}: {f}")

        seasons = self.loader.load_batch(files)
        return seasons

    def _transform_data(self, raw_seasons):
        """
        Takes raw season data and returns processed match data and standings
        """
        standings = self.transformer.batch(raw_seasons, self.transformer.get_standings)
        
        cleaned_seasons = self.transformer.batch(raw_seasons, lambda s: self.transformer.clean(s, COLS_TO_KEEP))

        processed_seasons, per_team_matches = self.transformer.transform(
            cleaned_seasons,
            standings
        )

        return processed_seasons, per_team_matches, standings

    def _save_data(self, processed_seasons, per_team_matches, standings):
        """
        Takes processed data and saves standings, processed match data (per team and input format) to specified directories
        """
        print(f"DEBUG: Saving {len(standings)} seasons starting from {STARTING_YEAR}")
        print(f"DEBUG: First team in standing: {standings[0]['Team'].iloc[0]}")
        print(f"DEBUG: length of standings {len(standings)}")

        self.writer.batch_save_to_dir(standings, STANDINGS_DIR, STARTING_YEAR, STANDINGS_PREFIX)

        self.writer.batch_save_to_dir(per_team_matches, PER_MATCH_PATH, STARTING_YEAR, 'per-team')
        self.writer.batch_save_to_dir(processed_seasons, PROCESSED_DATA_DIR, STARTING_YEAR, PROCESSED_PREFIX)

        inputs = self.transformer.concat_dfs(processed_seasons)
        self.writer.save_to_dir(inputs, PROCESSED_FULL_DATA_PATH, FULL_MATCH_DATA_NAME)

        train_df, val_df, test_df = self.transformer.get_splits(processed_seasons, 11, 2, 3)
        self.writer.save_to_dir(train_df, TRAIN_DATA_PATH, TRAIN_DATA_FILENAME)
        self.writer.save_to_dir(val_df, VAL_DATA_PATH, VAL_DATA_FILENAME)
        self.writer.save_to_dir(test_df, TEST_DATA_PATH, TEST_DATA_FILENAME)

    def run(self):
        # Get raw data with loader
        raw_seasons = self._get_raw_data()
        
        # Transform Data
        processed_seasons, per_team_matches, standings = self._transform_data(raw_seasons)

        # Save Processed Data
        self._save_data(processed_seasons, per_team_matches, standings)

        


if __name__ == "__main__":
    feature_types = [
        RollingWindowFeatures(window_size=WINDOW_SIZE),
        HeadToHeadFeatures(),
        PrevSeasonFeatures(
            cols_to_merge=PREV_SEASON_COLS, 
            baseline_pos=BASELINE_POS, 
            rename=RENAME_DICT
            )
    ]

    transformer = DataTransformer(feature_types)
    loader = Loader()
    writer = Writer()

    dp = DataPipeline(loader, transformer, writer)
    dp.run()