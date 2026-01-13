""" Class for cleaning dataset (creates season table and removes unwanted cols from data) """
import pandas as pd

class DataTransformer:
    def _convert_to_datetime(self, match_data: pd.DataFrame) -> pd.DataFrame:
        match_data["Date"] = pd.to_datetime(match_data["Date"], dayfirst=True)
        return match_data
    
    def _remove_cols(self, match_data: pd.DataFrame, target_cols: list):
        # here is where we keep certain columns
        return match_data[target_cols]
    
    def clean(self, match_data: pd.DataFrame) -> pd.DataFrame:
        match_data = self._convert_to_datetime(match_data)
        match_data = self._remove_cols(match_data)
        return match_data

    def add_form(self, season: pd.DataFrame):
        # need to compute form of each match by considering the x previous matches
        # for matches that are at the beginning of the season, we can 'bleed over' into the matches from the season prior
        pass

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
        return standings.sort_values("PTS", ascending=False)

    def rolling_form(self, season_matches: pd.DataFrame, per_team: pd.DataFrame, num_matches: int) -> pd.DataFrame:
        """ adds column for form over the last 5 matches for home and away team """
        season_matches["Form"] = per_team.rolling(num_matches).sum()
        return season_matches
       
    def build_per_team_matches(self, matches:pd.DataFrame) -> pd.DataFrame:
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
    
    def batch_build_per_team(self, seasons: list[pd.DataFrame]):
        pass