import pandas as pd
from abc import ABC, abstractmethod

class BaseFeatures(ABC):
    """
    Abstract Base Class for feature generators
    """
    def prepare(self, **kwargs):
        """ 
        Optional function to pass in parameters to concrete implementations

        Feature generators assign kwargs as required as class attributes
        """
        pass

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Logic for feature generation

        Transforms the input dataframe by adding columns for each added feature

        Args:
            df (pd.DataFrame): match data in team-match format
        
        Returns:
            pd.DataFrame: DataFrame with newly added features
        """
        pass

class RollingWindowFeatures(BaseFeatures):
    """ 
    Adds features based on recent match history, constrained by window size attribute

    Calculations involve rolling averages or sums over specified number of previous matches
    """
    def __init__(self, window_size: int):
        """ Initialises feature generator

        Args:
            window_size (int): Size of rolling window 
        """
        self.window_size = window_size

    def generate(self, df: pd.DataFrame):
        """ 
        Calculates rolling features and appends to provided DataFrame

        Args:
            df (pd.DataFrame): Input match data in match-team format
        
        Returns:
            pd.DataFrame: The original DataFrame object with newly added rolling window features
        """
        return df

class HeadToHeadFeatures(BaseFeatures):
    """
    Adds features based on the specific head to head matchup for each match

    Calculates sums and averages over specified number of previous matches against
    the specific opponent for a given match.

    Where data is not available, baseline statistics are assigned instead
    """
    def generate(self, df: pd.DataFrame):
        return df

class PrevSeasonFeatures(BaseFeatures):
    """ Class for adding features based on previous season's league standings """
    def __init__(self, cols_to_merge: list, baseline_pos: int, rename: dict):
        self.cols = cols_to_merge
        self.baseline_pos = baseline_pos
        self.rename = rename

    def prepare(self, prev_season: pd.DataFrame):
        self.prev_season = prev_season
    
    def generate(self, df: pd.DataFrame):
        """
        Adds fields from previous standings directly
        If previous standings do not exist, return df as is (ignore first season) 
        """
        if self.prev_season is None:
            for col in self.cols:
                df[f"prev_{col}"] = 0
            return df
        
        baseline = self.prev_season.loc[self.baseline_pos, self.cols]

        df = df.merge(
            self.prev_season[["Team"] + self.cols],
            left_on=["Team"],
            right_on=["Team"],
            how="left"
        )

        df[self.cols] = df[self.cols].fillna(baseline)
        df = df.rename(columns=self.rename)
        return df 