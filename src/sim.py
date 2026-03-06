""" Takes predicted expected goals and performs Monte Carlo simulation to generate WDL probabilities """
import math
import random

import numpy as np
import pandas as pd

from processing.loader import Loader
from processing.writer import Writer

from utils import PREDICTIONS_PATH, PROB_COLS, TEST_DATA_PATH, DATA_23, DATA_24, BET365, BET_365_C

class MatchSimulator:
    def __init__(self, n_sims: int):
        self.n_sims = n_sims
        
    def _sample_poisson(self, lmda: float) -> int:
        """ Takes a lambda (xG) and produces an integer by sampling from Poisson distribution"""
        L = math.exp(-lmda)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k-1

    def sim_match(self, row: pd.Series) -> tuple:
        """ Runs one simulation of a specific fixture

        Args:
            home_xg (int): expected goals for the home team
            away_xg (int): expected goals for the away team 

        Returns:
            int: 
        # """ 

        # home_xg, away_xg = row[0], row[1]

        # home_wins, draws, away_wins = 0, 0, 0

        # for i in range(self.n_sims):
        #     h_goals = self._sample_poisson(home_xg)
        #     a_goals = self._sample_poisson(away_xg)

        #     if h_goals > a_goals:
        #         home_wins += 1
        #     elif h_goals < a_goals:
        #         away_wins += 1
        #     else:
        #         draws += 1

        # home_win_prob = home_wins / self.n_sims
        # draw_prob = draws / self.n_sims
        # away_win_prob = away_wins / self.n_sims

        # return home_win_prob, draw_prob, away_win_prob



        h_lambda, a_lambda = row[0], row[1]

        home_goals = np.random.poisson(h_lambda, self.n_sims)
        away_goals = np.random.poisson(a_lambda, self.n_sims)

        home_win_prob = (np.sum(home_goals > away_goals) / self.n_sims).item()
        draw_prob = (np.sum(home_goals == away_goals) / self.n_sims).item()
        away_win_prob = (np.sum(home_goals < away_goals) / self.n_sims).item()

        return home_win_prob, draw_prob, away_win_prob
    
    def run_sim(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """ Takes matches DataFrame and adds column for W/D/L probabilities """
        probs = predictions.apply(self.sim_match, axis=1, result_type='expand')
        probs.columns = PROB_COLS
        predictions = pd.concat([predictions, probs], axis=1)
        return predictions
    
    def join_predictions(self, predictions: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
        """ Joins predictions on matches to get actual match outcomes """
        predictions["home_win"] = matches["W_home"]
        predictions["draw"] = matches["D_home"]
        predictions["away_win"] = matches["L_home"]
        return predictions

    def brier_score(self, df: pd.DataFrame) -> int:

        """Performance metric for results of Monte-Carlo simulation

        Measures mean squared difference between predicted probability assigned to possible
        outcomes for a given "row" or match, and the actual outcome of that match     
        
        BS = 1/N sum 1->N (ft - ot)^2

        Where ft is forecasted probability and ot is outcome of event at instance t (0 if it doesnt happen, one if it does)
        N is number of rows

        ^ above is only for single category

        Multi category Brier Score is given by:

        BS = 1/N sum 1->N sum 1->R (fti - oti)^2

        R is the number of possible classes in which the event can fall, and N is the overall number of instances of all classes.
        fti is predicted probability of class i
        oti is 1 if it is the ith class instance t; 0 otherwise. 

        in our case, R = 3

        We have all of our predictions already as a dataframe, do we need a separate class for the brier? probably not

        Args:
            predictions (pd.DataFrame): match data including probabiltiy distribution over home win / draw / away win

        Returns:
            int: Brier Score for predictions
        """

        R = 3 # R is number of categories of probability distribution / outcomes
        N = len(df)

        pred_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
        label_cols = ["home_win", "draw", "away_win"]

        df['BS'] = ((df[pred_cols].values - df[label_cols].values)**2).sum(axis=1)
        mean_BS = df['BS'].mean()

        return mean_BS, df


        
        

if __name__ == "__main__":
    # test for sim
    # first thing would be to load the predictions
    loader = Loader()
    writer = Writer()

    preds = loader.load(PREDICTIONS_PATH)

    # we have predictions, now time to get our probabilities
    simulator = MatchSimulator(20000)

    df = simulator.run_sim(preds)

    home = df.query("home_win_prob > draw_prob and home_win_prob > away_win_prob").shape[0]
    draws = df.query("draw_prob > home_win_prob and draw_prob > away_win_prob").shape[0]
    away = df.query("away_win_prob > draw_prob and away_win_prob > home_win_prob").shape[0]
    print(f"found {draws} matches where draw was most likely")
    print(f"found {home} matches where home is most likely")
    print(f"found {away} matches where away is most likely")


    matches = loader.load(TEST_DATA_PATH)

    # going to test if we can actually do the join correctly

    joined = simulator.join_predictions(df, matches )


    # now that we have the joined probabilities with the home_win, draw, away_win
    # we need to measure the brier score

    score, df = simulator.brier_score(joined) 

    print(f"Brier Score for predictions from 2023-2025: {score:.4f}")


    # would like to compare how my predictions shape up compared to the betting agencies
    # this gives a rough idea of how good the predictions actually are


    

    # writer.save_to_dir(df, "./probabilites", "MC.csv")



    # now that we have the probability distributions, we want to somehow "score"
    # the performance of the model
    
    bookies_probs_23 = loader.load(DATA_23)
    bookies_probs_24 = loader.load(DATA_24)



    bookies_probs = pd.concat([bookies_probs_23, bookies_probs_24], axis=0)
    bookies_probs = bookies_probs[BET_365_C]
    bookies_probs.columns = PROB_COLS

    bookies_probs[PROB_COLS] = 1 / bookies_probs[PROB_COLS]

    bookies_probs[PROB_COLS] = bookies_probs[PROB_COLS].div(bookies_probs[PROB_COLS].sum(axis=1), axis=0)
    print(bookies_probs.iloc[0:10])

    new_joined = simulator.join_predictions(bookies_probs, matches)

    new_score, other_df = simulator.brier_score(new_joined)

    print(f"BET365 Brier Score: {new_score:.4f}")

    
    # can use the Brier score



