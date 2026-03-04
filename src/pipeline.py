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

        stacked_short = self.transformer.concat_dfs(processed_seasons)

        # save stacked dataframe in short format
        self.writer.save_to_dir(stacked_short, 
                                self.config.path("short_stacked_dir"),
                                self.config.short_stacked_filename
                                )
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