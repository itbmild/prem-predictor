PROCESSED_FULL_DATA_PATH = './data/processed/processed_full.csv'
PROCESSED_MATCH_DATA_DIR = './data/processed/processed_yearly'
RAW_DATA_DIR = './data/raw'
LEAGUE_STANDINGS_DIR = './data/processed/standings_yearly'
STARTING_YEAR = 2008
PREM_FEATURES = ["HomeForm", "AwayForm", "HomePrevPTS", "AwayPrevPTS"]
PREM_LABELS = ["FTHG", "FTAG"]

TRAIN_DATA_PATH = "./data/processed/training/train_set.csv"
VAL_DATA_PATH = "./data/processed/validation/val_set.csv"
TEST_DATA_PATH = "./data/processed/test/test_set.csv"
MODEL_PATH = "./trained_model/NN_trained.pth"
PREDICTIONS_PATH = "./data/predictions/test_outputs.csv"

# constants for Eval
PREDICTION_COLS = ["home_xg", "away_xg"]
PROB_COLS = ["home_win_prob", "draw_prob", "away_win_prob"]

# constants for PrevSeasonFeatures
PREV_SEASON_COLS = ["PTS", "W", "D", "L", "GF", "GA", "SSN_YC", "SSN_RC", "AVG_SOT"]
BASELINE_POS = 17 # position of team from previous season to use for promoted teams
RENAME_DICT = {"PTS": "prevPTS"}

# constants for RollingWindowFeatures
WINDOW_SIZE = 5
TARGET_NAME_PAIRS = [("formPTS", "Form"), ("YC", "AVG_YC"), ("RC", "AVG_RC")]

# constants for HeadToHeadFeatures
BASELINE_PTS = 1.5 # pts assigned when match data is insufficient
MATCHES_THRESHOLD = 1
H2H_PAIRS = [("formPTS", "formH2H"), ("Goals", "avg_H2H_goals")]

# need a constant for things to directly pull from the raw data, i.e. the col names
COLS_TO_KEEP = [
    "Date", "HomeTeam", "FTHG", "HS", "HST", "HY", "HR", "HF",
    "AwayTeam", "FTAG", "AS", "AST", "AY", "AR", "AF",
    "FTR"
]

# bookies

XBET = ["1XBH", "1XBD", "1XBA"]
BET365 = [""]
