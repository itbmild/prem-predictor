"""
Contains Class to preprocess and load data for premier league prediction model
Creates dataset ready for input into ML model:

Raw data looks like:
Div
Date
HomeTeam
AwayTeam
FTHG : Full time home goals
FTAG : Away Goals
FTR : Full time Result
HTHG : Half time home team goals
HTAG : Half Time away goals
HTR : Half time result
HS : Home shots
AS : Away team shots
HST : Home team shots on target
AST : Away Team shots on target
HF : Home team fouls committed
AF : Away team fouls committed
HCAC : 
HY
AY
HR
AR
-------------------------------------------------------------
|             Home            |             Away            | 
| HLSP | HLSGF | HLSGA | HLSF | ALSP | ALSGF | ALSGA | ALSF |
-------------------------------------------------------------

PROBLEM
the model outputs have to match the inputs, i.e. they need to be able to produce a "match"
so that i can update rolling averages

i.e. if we choose to use some sort of results window to see next winner, we need a way to produce stats
on our next match, to then update the input that we are feeding into the NN.
easiest way to do this is to use same approach that we did for our latent representation of xG (lambda)
and do it for the other metrics that we are usnig to calculate?

For example, if we have a metric that is the form over the last 5 games, we are only concerned with
win/loss/draw

PROBLEM
the rows in the dataset that we are feeding the model are NOt the same as the rows from the raw dataset,
but we are using the metrics from the raw dataset and the features match up 1-1 with the rows.

IDEA
Create a dataframe that is empty, iterate through the processed dataset to produce the rows for the 
feature dataset.

i.e. for first row in our processed raw data, create a row in our input dataset that uses calculations
For example:
create row with (col1=last5games(team)) 


USEFUL
need a CSV file for the table finishing stats for each team, i.e.

Team Points GF GA GD W D L

How? use the raw data
Need a way to query the data, for example, get all games where hometeam = man city
"""

import os
import pandas as pd
from utils import PROCESSED_FULL_DATA_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR, LEAGUE_STANDINGS_DIR, STARTING_YEAR

class DataProcessor:
    COLS_TO_KEEP = [
                        "Date", "HomeTeam", "AwayTeam", 
                        "FTHG", "FTAG", "HS", "AS",
                        "HST", "AST"
        ]
    
    def __init__(self):
        # need to load the data if we havent already loaded and processed
        if os.path.isfile(PROCESSED_FULL_DATA_PATH):
            # load the csv file using pandas
            self.full_data = pd.read_csv(PROCESSED_FULL_DATA_PATH)
            self.yearly_data = self._read_yearly_csv()
        else:
            # instantiate list of dataframes, one for each season
            self.raw_data_frames = self._get_season_frames()
            self.process_data()

    ## HELPER FUNCTIONS
    def process_data(self):
        self._save_season_standings_csv()
        # processes raw data and saves CSVs for yearly and aggregated
        self._process_seasons()
        
        self._process_full()
        
    def _process_full(self):
        # remove unwanted columns from raw data and save to CSV file
        # # concatenate seasons into one dataframe
        # df_full = pd.concat(self.raw_data_frames)
        # print(df_full)
        pass
        

    def _process_seasons(self):
        year = STARTING_YEAR
        for i in self.raw_data_frames:
            # keep frames we want
            season = i[self.COLS_TO_KEEP]
            season.to_csv(os.path.join(PROCESSED_DATA_DIR, f'match-data-{year}.csv'))
            year += 1

    def _read_yearly_csv(self):
        pass

    """ Uses raw data for each season to create end of season table """
    def _save_season_standings_csv(self):
        """
        Takes raw data from all seasons, then saves
        table results as a CSV files at specified directory in utils
        """
        year = STARTING_YEAR
        for i in self.raw_data_frames:
            season_df = self._create_season_dataframe(i)
            season_df.to_csv(os.path.join(LEAGUE_STANDINGS_DIR, f'league-standings-{year}.csv'))
            year += 1
 
    def _create_season_dataframe(self, season: pd.DataFrame) -> pd.DataFrame:
        """
        Takes individual season raw data as a dataframe and returns
        dataframe containing the final standings of that season
        """
        home_stats = season.groupby("HomeTeam").agg(
            MP =("FTHG", "count"),
            W = ("FTR", lambda x: (x == 'H').sum()),
            D = ("FTR", lambda x: (x == 'D').sum()),
            L = ("FTR", lambda x: (x == 'A').sum()),
            GF = ("FTHG", "sum"),
            GA = ("FTAG", "sum")
        )
        away_stats = season.groupby("AwayTeam").agg(
            MP = ("FTAG", "count"),
            W = ("FTR", lambda x: (x == 'A').sum()),
            D = ("FTR", lambda x: (x == 'D').sum()),
            L = ("FTR", lambda x: (x == 'H').sum()),
            GF = ("FTAG", "sum"),
            GA = ("FTHG", "sum")
        )
        standings = home_stats + away_stats
        standings["GD"] = standings["GF"] - standings["GA"]
        standings["PTS"] = standings["W"] * 3 + standings["D"] * 1
        return standings.sort_values("PTS", ascending=False)

    def _clear_dir(self, dir):
        # iterates through directory and deletes all of the files
        contents = os.listdir(dir)
        for item in contents:
            os.remove(os.path.join(dir, item))

    def _get_season_frames(self) -> list[pd.DataFrame]:
        frames = []
        seasons = os.listdir(RAW_DATA_DIR)
        for season in seasons:
            # need to append a new dataframe for each season
            frames.append(pd.read_csv(os.path.join(RAW_DATA_DIR, season)))
        return frames
    
if __name__ == "__main__":
    proc = DataProcessor()