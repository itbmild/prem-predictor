""" CLI, entrypoint for Pipeline """

"""
can use argparse to setup CLI
need clear image of how the pipeline works
    
Data pipeline:
    
LOADER(Raw Data) -> Pipeline -> refined matches
    
Training
refined matches -> train/val/test splits
train -> Trainer(train) -> model.pth -> Writer(model.pth) 
Model is saved at specified directory
    
Evaluation
Loader(test data)? -> Model(test_data)
OR (depending on whether we have saved already or if we are going off previous loop)
if neither, then send an error message along the lines of "no saved model"
test data -> Model(test_data) -> simulate(output) -> compute brier score

by this point, we have probabilities for all of the matches, and we have the brier score
can compute additional metrics? i.e. direct xG comparison accuracy (w/ threshold)


##### structure of the main file
current structure is

DataPipeline   | train              | eval
calls run      | reads files        | loads model
processes      | instantiates model | runs sim
dumps in files | trains model       | reports results
"""
import argparse
import yaml
import pandas as pd
from config import Config
from processing.loader import Loader
from processing.writer import Writer
from processing.transform import DataTransformer
from processing.features import RollingWindowFeatures, HeadToHeadFeatures, PrevSeasonFeatures
from pipeline import DataPipeline




class PipelineOrchestrator:
    """ Orchestrator class for data processing / model training / model evaluation """
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = Config(yaml.safe_load(f))

        self.loader = Loader()
        self.writer = Writer()
        self.transformer = self._create_transformer()

    def _create_transformer(self):
        """
        Creates transformer object for adding features to the data
        """
        features = self.config['features']
        feature_types = [
            RollingWindowFeatures(
                window_size=features['window_size'],
                target_name_pairs=features['target_name_pairs']
            ),
            PrevSeasonFeatures(
                cols_to_merge=features['previous_cols'],
                baseline_pos=features['baseline'],
                rename=features['rename']
            )
        ]
        combined_features = [
            HeadToHeadFeatures(
                window_size=features['window_size'],
                target_name_pairs=features['target_name_pairs']
            )
        ]
        return DataTransformer(feature_types, combined_features)
    
    def process_data(self):
        """ Takes raw data and processes it to prepare for model input """
        pl = DataPipeline(self.loader, self.transformer, self.writer, self.config.pipeline)
        pl.run()

    def train_model(self):
        pass

    def eval_model(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Premier League Predictor")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # Data Pipeline command
    prepare_parser = subparsers.add_parser("prepare", help="Pre-processese raw match data")
    prepare_parser.add_argument("--input_dir", type=str, default="./data/raw", help="Path to folder containing the raw CSV data")
    prepare_parser.add_argument("--save-to", type=str, default="./data/cleaned")
    args = parser.parse_args()
    # need to instantiate a loader and writer
    if args.command == "prepare":
        # Data pipeline logic goes in here 
        # we basically just want to store the cleaned data as one big stacked up DF.
        # we can do the train/val/test split later on based on the information that each season
        pass
    else:
        print("how did you get here?")

def load_data(self) -> pd.DataFrame:
    pass
    # return raw_df


def test():
    # runs a test example thingamabob
    orc_path = "./config.yaml"


    pl = PipelineOrchestrator(orc_path)
    pl.process_data()


if __name__ == "__main__":
    test()
    # main()

