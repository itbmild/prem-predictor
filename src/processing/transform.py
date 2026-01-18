""" Class for cleaning dataset (creates season table and removes unwanted cols from data) """
import pandas as pd
from _collections_abc import Callable

class DataTransformer:
    def _convert_to_datetime(self, match_data: pd.DataFrame) -> pd.DataFrame:
        match_data["Date"] = pd.to_datetime(match_data["Date"], format="mixed", dayfirst=True)
        return match_data
    
    def _remove_cols(self, match_data: pd.DataFrame, target_cols: list):
        # here is where we keep certain columns
        return match_data[target_cols]
    
    def clean(self, match_data: pd.DataFrame, target_cols: list) -> pd.DataFrame:
        match_data = self._convert_to_datetime(match_data)
        match_data = self._remove_cols(match_data, target_cols)
        return match_data

    def add_form(self, per_team: pd.DataFrame, num_matches: int) -> pd.DataFrame:
        """ takes per team data and adds the form of that match"""
        per_team = per_team.sort_values(['Team', 'Date'])
        rolling_pts = (
            per_team.groupby("Team")["PTS"]
            .rolling(window=num_matches, min_periods=1)
            .sum()
            .values
        )
        per_team["_tmp_rolling"] = rolling_pts
        per_team["Form"] = per_team.groupby("Team")["_tmp_rolling"].shift(1)
        return per_team.drop(columns=["_tmp_rolling"]).fillna(0).reset_index(drop=True)
       
    def add_pts(self, season_matches: pd.DataFrame, season_table: pd.DataFrame) -> pd.DataFrame:
        """ append previous seasons points to each match for both home and away team """
        # need to find a way to query the league table from the season before
        season_matches = season_matches.merge(season_table[["Team", "PTS"]],
                                    left_on="HomeTeam",
                                    right_on="Team",
                                    how="left"
                                    ).rename(columns={"PTS": "home_prev_points"}) \
                                    .drop(columns=["Team"])
        
        return season_matches.merge(season_table[["Team", "PTS"]],
                                              left_on="AwayTeam",
                                              right_on="Team",
                                              how="left"
                                              ).rename(columns={"PTS": "away_prev_points"}) \
                                              .drop(columns=["Team"])

    def get_standings(self, season: pd.DataFrame) -> pd.DataFrame:
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
        standings = standings.reset_index().rename(columns={"HomeTeam" : "Team"})
        standings["GD"] = standings["GF"] - standings["GA"]
        standings["PTS"] = standings["W"] * 3 + standings["D"] * 1
        standings = standings.sort_values("PTS", ascending=False).reset_index(drop=True)
        standings["Position"] = standings.index + 1
        return standings

    def rolling_form(self, season_matches: pd.DataFrame, per_team: pd.DataFrame, num_matches: int) -> pd.DataFrame:
        """ adds column for form over the last 5 matches for home and away team """
        season_matches["Form"] = per_team.rolling(num_matches).sum()
        return season_matches
       
    def build_per_team(self, matches:pd.DataFrame) -> pd.DataFrame:
        """ Takes match data and produces dataframe to allow for easier aggregation of features """
        home = pd.DataFrame({
            "Date": matches["Date"],
            "Team": matches["HomeTeam"],
            "Opponent": matches["AwayTeam"],
            "PTS": matches["FTR"].map({"H": 3, "D": 1, "A": 0}),
            "isHome": 1
        })
        away = pd.DataFrame({
            "Date": matches["Date"],
            "Team": matches["AwayTeam"],
            "Opponent": matches["HomeTeam"],
            "PTS": matches["FTR"].map({"H": 0, "D": 1, "A": 3}),
            "isHome": 0
        })
        return pd.concat([home,away], ignore_index=True)
    
    def get_batch_standings(self, seasons: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """ takes list of raw match data and generates end-of-season table results """
        standings = []
        for season in seasons:
            table = self.get_standings(season)
            standings.append(table)
        return standings
    
    def batch_clean_seasons(self, seasons: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """ Takes list of raw match data for each season and returns cleaned list """
        clean_seasons = []
        for season in seasons:
            cleaned = self.clean(season)
            clean_seasons.append(cleaned)
        return clean_seasons
    
    def batch_build_per_team(self, seasons: list[pd.DataFrame]) -> list[pd.DataFrame]:
        per_team_seasons = []
        for season in seasons:
            per_team = self.build_per_team(season)
            per_team_seasons.append(per_team)
        return per_team_seasons
    
    def batch(self, list: list, callable: Callable):
        """ 
        takes atomic transformation method and applies to input list
        returns accumulated transformed list
        """
        acc = []
        for item in list:
            atom = callable(item)
            acc.append(atom)
        return acc
    
    def merge_form(self, matches: pd.DataFrame, per_team: pd.DataFrame) -> pd.DataFrame:
        join_on = per_team[["Date", "Team", "Form"]]
        matches = matches.merge(
            join_on,
            left_on=["Date", "HomeTeam"],
            right_on=["Date", "Team"],
            how="left"
        ).rename(columns={"Form": "HomeForm"}).drop("Team", axis=1)

        matches = matches.merge(
            join_on,
            left_on=["Date", "AwayTeam"],
            right_on=["Date", "Team"],
            how="left"
        ).rename(columns={"Form": "AwayForm"}).drop("Team", axis=1)
        return matches

    def merge_points(self, matches: pd.DataFrame, table: pd.DataFrame):
        """ takes in a season and merges the teams previous points """
        join_on = table[["Team", "PTS"]]
        matches = matches.merge(
            join_on,
            left_on=["HomeTeam"],
            right_on=["Team"],
            how="left"
        ).rename(columns={"PTS": "HomePrevPTS"}).drop("Team", axis=1)

        matches=matches.merge(
            join_on,
            left_on=["AwayTeam"],
            right_on=["Team"],
            how="left"
        ).rename(columns={"PTS": "AwayPrevPTS"}).drop("Team", axis=1)
        return matches

    def merge_position(self, matches: pd.DataFrame, table: pd.DataFrame):
        """ 
            Takes matches from a given season, and a league table from the previous season
            Adds column to match data containing previous league table finish
            Clubs promoted in given season are given points from 18th place team in prev season
        """
        join_on = table[["Team", "Position"]] 
        # merge home team position
        matches = matches.merge(
            join_on,
            left_on=["HomeTeam"],
            right_on=["Team"],
            how="left"
        ).rename(columns={"Position": "HomePrevPos"}).drop("Team", axis=1)

        # merge away team position
        matches = matches.merge(
            join_on,
            left_on=["AwayTeam"],
            right_on=["Team"],
            how="left"
        ).rename(columns={"Position": "AwayPrevPos"}).drop("Team", axis=1)
        return matches

    def add_features(self, seasons, per_team, standings) -> pd.DataFrame:
        """ 
        Takes match data and aggregates data from supplementary standings and per_team tables
        To produce features
        """
        # current implementation iterates through each season
        # WE DONT WANT THIS
        # WHAT WE WANT:
        # ignore the first season, and only apply aggregation to second onwards
        # how do we do this 

        for i in range(1,len(seasons)):
            # we need to skip the first season
            
            season = seasons[i]
            team_season = per_team[i]
            table = standings[i-1]

            # last X games features
            season = self.merge_form(season, team_season)

            # last season features
            season = self.merge_points(season, table)
            season = self.merge_position(season, table)
            seasons[i] = season
        return seasons

    def concat_dfs(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """ Takes list of dataframes and returns concatentation of list """
        return pd.concat(dfs, axis=0, ignore_index=True)
        
    def get_splits(self, dfs: list[pd.DataFrame], train_samples: int, val_samples: int, test_samples: int):
        train_set = pd.concat(dfs[1:train_samples+1])
        val_set = pd.concat(dfs[:train_samples+1:val_samples+train_samples])
        test_set = pd.concat(dfs[val_samples+train_samples:])

        print(len(train_set))
        return (train_set, val_set, test_set)