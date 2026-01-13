""" 
    Data pipeline
    Processes raw data and saves to files with defined loading, cleaning, feature and writer classes
"""
from processing.loader import Loader
from processing.writer import Writer
from processing.features import FeatureTransformer
from processing.transform import DataTransformer
import pandas as pd

COLS_TO_KEEP = [
                        "Date", "HomeTeam", "AwayTeam", 
                        "FTHG", "FTAG", "HS", "AS",
                        "HST", "AST", "FTR"
        ]

class DataPipeline:
    def __init__(self, loader: Loader, transformer: DataTransformer, writer: Writer):
        self.loader = loader
        self.transformer = transformer
        self.writer = writer

    def run(self):
        """ Runs full data pipeline """
        # load match data from files
        files = self.loader.get_files()
        seasons = self.loader.load_batch(files)

        standings = self.transformer.batch(seasons, self.transformer.get_standings)
        seasons = self.transformer.batch(seasons, lambda s: self.transformer.clean(s, COLS_TO_KEEP))
        per_team = self.transformer.batch(seasons, self.transformer.build_per_team)


        
        per_team[0] = self.transformer.add_form(per_team[0], 5)

        ###
        # need to put the form into each match somehow, i.e.
        # matches -> go to the per_team col and find that specific match -> add form into our match
        # merge form accepts the raw match data from one season, and adds the home form for each match



        example_matches = seasons[0]

        # print("--------------------- BEFORE MERGE ---------------------")
        # print(example_matches)
        print(per_team[0].query("Team == 'Sunderland'"))

        example_matches = self.transformer.merge_form(example_matches, per_team[0])
        print("------------------- AFTER MERGE ---------------------")
        print(example_matches[30:40])

        ###

if __name__ == "__main__":
    transformer = DataTransformer()
    features = FeatureTransformer()
    loader = Loader('./data/raw')
    writer = Writer()

    dp = DataPipeline(loader, transformer, writer)
    dp.run()