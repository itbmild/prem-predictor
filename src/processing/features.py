import pandas as pd
from abc import ABC, abstractmethod
from utils import MATCHES_THRESHOLD

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
    def __init__(self, window_size: int, target_name_pairs: list[tuple]):
        """ Initialises feature generator

        Args:
            window_size (int): Size of rolling window 
        """
        self.window_size = window_size
        self.pairs = target_name_pairs

    def generate(self, df: pd.DataFrame):
        """ 
        Calculates rolling features and appends to provided DataFrame

        Args:
            df (pd.DataFrame): Input match data in match-team format
        
        Returns:
            pd.DataFrame: The original DataFrame object with newly added rolling window features

            Average Goals
            Average Goals against
            Average Shots on target

            X Average Points
            X Average Ycards
            X Average RCards
            
            Wins
            Losses
            Draws

            need a way to calculate each of these over the last X amount of games given team-match format
            we have:
            Date, Team, Opponent, ... 
            # need to roll over previous x games from that same team
        """

        for (target, name) in self.pairs:
            # need to compute the averages for all these pairs
            df = self._rolling_avg(df, target, name)
        return df
    
    def _rolling_avg(self, df: pd.DataFrame, target: str, name: str) -> pd.DataFrame:
        """
        Computes the average over the specified column over previous recent matches and adds as feature
        
        Args:
            df (pd.DataFrame): Input match data in team-match format
        
        Returns:
            pd.DataFrame: Original DataFrame object with average col added
        """
        df = df.sort_values(['Team', 'Date'])

        shift = df.groupby('Team')[target].shift(1)

        df[name] = (
            shift
            .groupby(df['Team'])
            .rolling(window=self.window_size, min_periods=MATCHES_THRESHOLD)
            .mean()
            .values
        )

        df[name] = df[name].fillna(0)
        df = df.sort_values('Date')
        
        return df

    def _rolling_sum(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

def debug_per_team(df, team_name):
    # Filter for the specific team
    team_df = df[df['Team'] == team_name].copy()
    
    # Ensure it is chronological
    team_df = team_df.sort_values('Date')
    
    print(f"\n--- Checking {team_name} Logic ---")
    print(team_df.to_string(index=False))
    
    # Quick math check for the user
    last_5_actual = team_df['formPTS'].tail(5).mean()
    print(f"\nFinal Calculated Mean of last 5: {last_5_actual:.2f}")
    
class HeadToHeadFeatures(BaseFeatures):
    """
    Adds features based on the specific head to head matchup for each match

    Calculates sums and averages over specified number of previous matches against
    the specific opponent for a given match.

    Where data is not available, baseline statistics are assigned instead
    """
    def __init__(self, window_size: int, target_name_pairs: list[tuple]):
        self.window_size = window_size
        self.pairs = target_name_pairs

    def generate(self, df: pd.DataFrame):
        """Generates features based on specific head to head matchup of given match

        Args:
            df (pd.DataFrame): DataFrame of all seasons combined

        Returns:
            pd.DataFrame: original DataFrame object with H2H feature added 
        """
        for target, name in self.pairs:
            df = self._H2H_rolling_avg(df, target, name)

        test_h2h_logic(df)
        return df

    
    def _H2H_rolling_avg(self, df: pd.DataFrame, target: str, name: str):
        df = df.sort_values(['Team', 'Opponent', 'Date'])
        shift = df.groupby(['Team', 'Opponent'])[target].shift(1)

        df[name] = (
            shift
            .groupby([df['Team'], df['Opponent']])
            .rolling(window=self.window_size, min_periods=MATCHES_THRESHOLD)
            .mean()
            .values
        )

        df[name] = df[name].fillna(0)
        df = df.sort_values('Date')
        return df
    
def test_h2h_logic(df):
    # 1. Filter for one specific matchup
    # We want to see Arsenal's performance specifically against Man City
    test_case = df[(df['Team'] == 'Arsenal') & (df['Opponent'] == 'Man City')].copy()
    
    # 2. Sort by date to see the timeline
    test_case = test_case.sort_values('Date')
    
    # 3. Display the key columns
    print("Verifying H2H Form: Arsenal vs Man City")
    cols_to_show = ['Date', 'Team', 'Opponent', 'Goals', 'avg_H2H_goals']
    print(test_case[cols_to_show])

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
            # for col in self.cols:
            #     df[f"prev_{col}"] = 0
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