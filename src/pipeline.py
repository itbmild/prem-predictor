""" 
    Data pipeline
    Processes raw data and saves to files with defined loading, cleaning, feature and writer classes
"""
from processing.loader import Loader
from processing.writer import Writer
from processing.transform import DataTransformer
import pandas as pd

class DataPipeline:
    def __init__(self, loader: Loader, transformer: DataTransformer, writer: Writer, config: dict=None):
        self.loader = loader
        self.transformer = transformer
        self.writer = writer
        self.config = config

    def _get_raw_data(self) -> pd.DataFrame:
        """
        Loads raw match data from file and returns pandas dataframe containing the data
        """
        files = self.loader.get_files(self.config.path("raw_data_dir"))
        seasons = self.loader.load_batch(files)
        return seasons

    def _transform_data(self, raw_seasons):
        """
        Takes raw season data and returns processed match data and standings
        """
    
        standings = self.transformer.batch(raw_seasons, self.transformer.get_standings)
        cleaned_seasons = self.transformer.batch(raw_seasons, lambda s: self.transformer.clean(s, self.config.cols))
        processed_seasons, per_team_matches = self.transformer.transform(
            cleaned_seasons,
            standings
        )

        return processed_seasons, per_team_matches, standings

    def _save_data(self, processed_seasons, per_team_matches, standings):
        """
        Takes processed data and saves standings, processed match data (per team and input format) to specified directories
        """
        # save each individual season table
        self.writer.batch_save_to_dir(standings,
                                      self.config.path("standings_dir"),
                                      self.config.starting_year,
                                      self.config.standings_prefix
                                      )
        
        # save each seasons matches in team-match format
        self.writer.batch_save_to_dir(per_team_matches,
                                      self.config.path("team_match_dir"),
                                      self.config.starting_year,
                                      self.config.team_match_prefix
                                      )
        
        # save each processed season in short format
        self.writer.batch_save_to_dir(processed_seasons,
                                      self.config.path("yearly_dir"),
                                      self.config.starting_year,
                                      self.config.yearly_prefix
                                      )

        # save stacked processed seasons (short format)
        stacked_short = self.transformer.concat_dfs(processed_seasons)
        self.writer.save_to_dir(stacked_short, 
                                self.config.path("short_stacked_dir"),
                                self.config.short_stacked_filename
                                )
        
        # save stacked processed seasons (team-match format)
        stacked_team_match = self.transformer.concat_dfs(per_team_matches)
        self.writer.save_to_dir(stacked_team_match,
                                self.config.path("team_match_stacked_dir"),
                                self.config.team_match_stacked_filename
                                )
        
        # save train/val/test splits in short format
        short_train, short_val, short_test = self._get_splits(stacked_short)
        self._save_splits(self.config.path("short_splits_dir") ,short_train, short_val, short_test)

        tm_train, tm_val, tm_test = self._get_splits(stacked_team_match)
        self._save_splits(self.config.path("team_match_splits_dir"),tm_train, tm_val, tm_test)

    def _save_splits(self, dir, train, val, test):
        self.writer.save_to_dir(train, dir, "train")
        self.writer.save_to_dir(val, dir, "val")
        self.writer.save_to_dir(test, dir, "test")


        





        

    def _get_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Takes stacked dataframe (in short or team-match format) and returns train/val/test dfs """
        splits = self.config.splits
        train_p, val_p, test_p = splits[0], splits[1], splits[2]

        if train_p + val_p + test_p != 1:
            raise ValueError(f"train/val/test split proportions must sum to 1. Instead summed to {train_p + val_p + test_p}")

        
        start_idx = self.config.season_length
        rows = len(df) - start_idx

        # need to find the cutoff indices for each ting
        train_cutoff = start_idx + int(train_p * rows)
        val_cutoff = int(val_p * rows) + train_cutoff
        print(train_cutoff)


        train_df = df[start_idx:train_cutoff]
        val_df = df[train_cutoff:val_cutoff]
        test_df = df[val_cutoff:rows]

        # print(len(train_df))
        # print(len(val_df))
        # print(len(test_df))

        return train_df, val_df, test_df
        

    def run(self):
        """
        Runs entire pipeline process
        """
        # Get raw data with loader
        raw_seasons = self._get_raw_data()
        # Transform Data
        processed_seasons, per_team_matches, standings = self._transform_data(raw_seasons)
        # Save Processed Data
        self._save_data(processed_seasons, per_team_matches, standings)