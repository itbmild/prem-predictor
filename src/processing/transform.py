""" Class for cleaning dataset (creates season table and removes unwanted cols from data) """
import pandas as pd
from _collections_abc import Callable
from .features import BaseFeatures

class DataTransformer:
    def __init__(self, per_season_steps: list[BaseFeatures]=[], combined_steps: list[BaseFeatures]=[]):
        self.per_season_steps = per_season_steps
        self.combined_steps = combined_steps

    def _convert_to_datetime(self, match_data: pd.DataFrame) -> pd.DataFrame:
        match_data["Date"] = pd.to_datetime(match_data["Date"], format="mixed", dayfirst=True)
        return match_data
    
    def _remove_cols(self, match_data: pd.DataFrame, target_cols: list):
        # here is where we keep certain columns
        return match_data[target_cols].copy()
    
    def clean(self, match_data: pd.DataFrame, target_cols: list) -> pd.DataFrame:
        match_data = self._convert_to_datetime(match_data)
        match_data = self._remove_cols(match_data, target_cols)
        return match_data

    def add_form(self, per_team: pd.DataFrame, num_matches: int) -> pd.DataFrame:
        """ takes per team data and adds the form of that match"""
        per_team = per_team.sort_values(['Team', 'Date'])
        rolling_pts = (
            per_team.groupby("Team")["formPTS"]
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
            GA = ("FTAG", "sum"),
            SSN_YC = ("HY", "sum"),
            SSN_RC = ("HR", "sum"),
            SOT = ("HST", "sum"),
            SH = ("HS", "sum")
        )
        away_stats = season.groupby("AwayTeam").agg(
            MP = ("FTAG", "count"),
            W = ("FTR", lambda x: (x == 'A').sum()),
            D = ("FTR", lambda x: (x == 'D').sum()),
            L = ("FTR", lambda x: (x == 'H').sum()),
            GF = ("FTAG", "sum"),
            GA = ("FTHG", "sum"),
            SSN_YC = ("AY", "sum"),
            SSN_RC = ("AR", "sum"),
            SOT = ("AST", "sum"),
            SH = ("AS", "sum")
        )

        standings = home_stats + away_stats
        standings = standings.reset_index().rename(columns={"HomeTeam" : "Team"})
        standings = self._standings_calc(standings)
        return standings

    def _standings_calc(self, standings: pd.DataFrame) -> pd.DataFrame:
         # calculations
        standings["GD"] = standings["GF"] - standings["GA"]
        standings["PTS"] = standings["W"] * 3 + standings["D"] * 1
        standings = standings.sort_values("PTS", ascending=False).reset_index(drop=True)
        standings["Position"] = standings.index + 1
        standings["AVG_SOT"] = (standings["SOT"] / standings["MP"])
        # need to add one for shots total
        standings["AVG_SH"] = (standings["SH"] / standings["MP"])
        return standings

    def rolling_form(self, season_matches: pd.DataFrame, per_team: pd.DataFrame, num_matches: int) -> pd.DataFrame:
        """ adds column for form over the last 5 matches for home and away team """
        season_matches["Form"] = per_team.rolling(num_matches).sum()
        return season_matches
       
    def _match_team_format(self, matches:pd.DataFrame) -> pd.DataFrame:
        """ Takes match data and produces dataframe to allow for easier aggregation of features """
        home = pd.DataFrame({
            "Date": matches["Date"],
            "Team": matches["HomeTeam"],
            "Opponent": matches["AwayTeam"],
            "formPTS": matches["FTR"].map({"H": 3, "D": 1, "A": 0}),
            "isHome": 1,
            "FTHG": matches["FTHG"],
            "FTAG": matches["FTAG"],
            "YC": matches["HY"],
            "RC": matches["HR"],
            "Goals": matches["FTHG"]
        })
        away = pd.DataFrame({
            "Date": matches["Date"],
            "Team": matches["AwayTeam"],
            "Opponent": matches["HomeTeam"],
            "formPTS": matches["FTR"].map({"H": 0, "D": 1, "A": 3}),
            "isHome": 0,
            "FTHG": matches["FTHG"],
            "FTAG": matches["FTAG"],
            "YC": matches["AY"],
            "RC": matches["AR"],
            "Goals": matches["FTAG"]
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
        # return self.join_home_away(
        #     matches,
        #     per_team,
        #     cols=["Date", "Team", "Form"],
        #     home_col="HomeForm",
        #     away_col="AwayForm"
        # )
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
        # let prev points be that of previous 18th place team
        baseline = table.loc[17, "PTS"] 
        return self.join_home_away(
            matches,
            table,
            cols=["Team", "PTS"],
            home_col="HomePrevPTS",
            away_col="AwayPrevPTS",
            fill=baseline
        )

    def merge_position(self, matches: pd.DataFrame, table: pd.DataFrame):
        """ 
            Takes matches from a given season, and a league table from the previous season
            Adds column to match data containing previous league table finish
            Clubs promoted in given season are given points from 18th place team in prev season
        """
        fill = 18 # assume finished at top of relegation zone for promoted sides
        return self.join_home_away(
            matches, 
            table, 
            cols=["Team", "Position"],
            home_col="HomePrevPos",
            away_col="AwayPrevPos",
            fill=fill
        )

    def concat_dfs(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """ Takes list of dataframes and returns concatentation of list """
        return pd.concat(dfs, axis=0, ignore_index=True)
        
    def get_splits(self, dfs: list[pd.DataFrame], train_samples: int, val_samples: int, test_samples: int):
        train_set = pd.concat(dfs[1:train_samples + 1])
        val_set = pd.concat(dfs[train_samples + 1:val_samples + train_samples + 1])
        test_set = pd.concat(dfs[1 + val_samples + train_samples : train_samples + val_samples+ test_samples + 1])

        return (train_set, val_set, test_set)

    def join_home_away(self, matches: pd.DataFrame, table: pd.DataFrame, cols: list, home_col: str, away_col: str, fill=None) -> pd.DataFrame:
        join_on = table[cols]
        # merge for Home team
        matches = matches.merge(
            join_on,
            left_on=["HomeTeam"],
            right_on=["Team"],
            how="left"
        ).rename(columns={cols[1]: home_col}).drop("Team", axis=1)

        # merge for away team
        matches = matches.merge(
            join_on,
            left_on=["AwayTeam"],
            right_on=["Team"],
            how="left"
        ).rename(columns={cols[1]: away_col}).drop("Team", axis=1)

        if fill is not None:
            matches[home_col] = matches[home_col].fillna(fill)
            matches[away_col] = matches[away_col].fillna(fill)
        return matches
    
    def join_current_season(self, matches: pd.DataFrame, per_match: pd.DataFrame):
        pass

    def add_features(self, team_matches: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        """
        Takes long form of matches (2 rows per match) and adds features for each team in each match
        Returns long form dataframe for all matches containing features.
        """
        team_matches = self.add_prev_pts(team_matches, table)
        team_matches = self.reformat_matches(team_matches)
        return team_matches

    def reformat_matches(self, team_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Takes matches in long format (2 rows per match) and formats in the form of 
        HomeTeam features + AwayTeam Features
        """
        cols_to_drop = [
            "Team_away", "Opponent_away",
            "isHome_home", "isHome_away",
            "FTHG_away", "FTAG_away"
        ]
        # we have a table of the form 
        # Date,Team,Opponent,formPTS,isHome,Form,prevPTS,W,D,L,GF,GA,YC,RC
        # we want to prefix home to all these, and prefix away for the away
        home_matches = team_matches[team_matches["isHome"] == 1].copy()
        away_matches = team_matches[team_matches["isHome"] == 0].copy()

        matches_full = home_matches.merge(
            away_matches,
            left_on=["Date", "Team", "Opponent"],
            right_on=["Date", "Opponent", "Team"],
            suffixes=["_home", "_away"]
        ).rename(columns={"team_home": "HomeTeam", "Opponent_home": "AwayTeam", "FTHG_home": "FTHG", "FTAG_home": "FTAG"})
        matches_full = matches_full.drop(columns=cols_to_drop)

        return matches_full

    def add_prev_pts(self, matches_long: pd.DataFrame, table: pd.DataFrame):
        """ 
        Takes matches in per_team format and adds column for points in previous season
        Teams that were promoted in the current season are assigned points of 18th place
        Team from previous season as a baseline
        """
        baseline = table[["PTS", "W", "D", "L", "GF", "GA", "YC", "RC", "AVG_SOT"]].loc[17]
        cols = ["PTS", "W", "D", "L", "GF", "GA", "YC", "RC", "AVG_SOT"]

        baseline_row = table.loc[17, cols]
        fill_dict = {col: baseline[col] for col in cols}

        matches_long = matches_long.merge(
            table[["Team", "PTS", "W", "D", "L", "GF", "GA", "YC", "RC", "AVG_SOT"]],
            left_on=["Team"],
            right_on=["Team"],
            how="left"
        )
        matches_long[cols] = matches_long[cols].fillna(baseline_row)
        matches_long = matches_long.rename(columns={"PTS": "prevPTS"})
        return matches_long

    def batch_add_features(self, team_matches_list: list[pd.DataFrame], tables_list: list[pd.DataFrame]):
        # add all previous season features
        for i in range(1, len(team_matches_list)):
            season = team_matches_list[i]
            previous_table = tables_list[i-1]
            season = self.add_features(season, previous_table)
            season = season.sort_values("Date", ascending=True)
            team_matches_list[i] = season
            
        # add aggregate features based on rolling window


        return team_matches_list
    
    def batch_add_WDL(self, matches_list: list[pd.DataFrame]) -> list[pd.DataFrame]:
        for i in range(len(matches_list)):
            match = matches_list[i]
            match = self.add_WDL(match)
            matches_list[i] = match


        return matches_list

    def add_WDL(self, matches: pd.DataFrame):
        # need to add W, D and L columns.

        matches["W_home"] = (matches["FTHG"] > matches["FTAG"]).astype(int)
        matches["D_home"] = (matches["FTHG"] == matches["FTAG"]).astype(int)
        matches["L_home"] = (matches["FTHG"] < matches["FTAG"]).astype(int)
        return matches

######################
######################
######################
    
    def transform(self, seasons: list[pd.DataFrame], standings: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Main entrypoint for processing, performs all necessary processing and adds features to data
        """
        # new logic for strategy pattern
        team_match_seasons = []

        for i, season_df in enumerate(seasons):
            curr_df = season_df.copy()

            # convert to match_team_format for generation
            curr_df = self._match_team_format(curr_df)
            prev_standings = standings[i-1] if i > 0 else None

            # for features constrained to one season
            for step in self.per_season_steps:
                 step.prepare(prev_season=prev_standings)
                 curr_df = step.generate(curr_df)

            team_match_seasons.append(curr_df)

        combined = self._combine(team_match_seasons)
        for step in self.combined_steps:
            # results in a combined df where we wanna re-perform the splitting?
            combined = step.generate(combined) 

        # resplit into per-season
        processed_seasons = []
        final_team_match = []
        for i, group in combined.groupby('season', sort=True):
            clean = group.drop(columns=['season'])
            final_team_match.append(clean)

            final_df = self.reformat_matches(clean)
            final_df = self.add_WDL(final_df)
            processed_seasons.append(final_df)


        return processed_seasons, final_team_match

        ####################
        seasons_processed = self.batch(seasons, lambda s: self.clean(s, COLS_TO_KEEP))

        # convert usual format to per team format 
        per_team_processed = self.batch(seasons_processed, self.build_per_team)

        # add rolling features and include previous standings dat

        per_team_processed = self.batch(per_team_processed, lambda s: self.add_form(s, 5))

        # we are calling add features, should dispatch the add form from there?
        per_team_processed = self.batch_add_features(per_team_processed, standings)
        per_team_processed = self.batch_add_WDL(per_team_processed)

        return seasons_processed, per_team_processed

    def _combine(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Takes list of seasons as DataFrames and combines them into one dataFrame

        Args:
            dfs (list[DataFrame]): list of processed season data in team-match format

        Returns:
            Combined list of season data with season as index 
        """
        for i, df in enumerate(dfs):
            df['season'] = i

        df = pd.concat(dfs, ignore_index=True)

        df = df.sort_values('Date')
        return df