# PREM_FEATURES = ["HomeForm", "AwayForm", "HomePrevPTS", "AwayPrevPTS"]
PREM_FEATURES = [
    "Form_home", "prevPTS_home", "W_home", "D_home", "L_home",
    "GF_home", "GA_home", "YC_home", "RC_home",
    "Form_away", "prevPTS_away", "W_away", "D_away", "L_away",
    "GF_away", "GA_away", "YC_away", "RC_away"
    ]
PREM_LABELS = ["FTHG", "FTAG"]

PREM_EVAL_LABELS = ["W", "D", "L"]

"""
need to add more features.

Types of features: From last season, last x games, against x opponent

From Last Season:
    X Position         
    X Wins
    X Draws
    X Losses
    X Total Points
    X Goals For
    X Goals Against
    X Red Cards
    X Yellow Cards
    Average Big Chances
    Average Shots Leading to Goals
    Average Shots on target
    Average Shots

Last X Games:
    Average Shots leading to Goals
    Average Shots on target
    # Average Points
    Average Ycards
    Average RCards
    Average Goals
    Average Goals against
    Wins
    Losses
    Draws

Against X team:
    Win Rate
    Goals For
    Goals Against
    Shots leading to goals
    Average Shots leading to goals
    Average Corners
    Average Points
    Average YCards
    Average RCards
    Average Goals
    Average Goals Against

"""