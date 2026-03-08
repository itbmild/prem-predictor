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
import torch
from config import Config
from processing.loader import Loader
from processing.writer import Writer
from processing.transform import DataTransformer
from processing.features import RollingWindowFeatures, HeadToHeadFeatures, PrevSeasonFeatures
from models.dataset import PLDataModule, PLDataset
from pipeline import DataPipeline
from models.trainer import NNTrainer

from models.modules import NeuralNet

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pathlib import Path

class PipelineOrchestrator:
    """ Orchestrator class for data processing / model training / model evaluation """
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = Config(yaml.safe_load(f))

        self.loader = Loader()
        self.writer = Writer()
        self.transformer = self._create_transformer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                target_name_pairs=features['h2h_pairs']
            )
        ]
        return DataTransformer(feature_types, combined_features, self.config.transformer)
    
    def process_data(self):
        """ Takes raw data and processes it to prepare for model input """
        pl = DataPipeline(self.loader, self.transformer, self.writer, self.config.pipeline)
        pl.run()

    def train(self, model: str):
        """
        Runs training on specified model type and saves to 
        directory specified in config.yaml
        """
        model_config = self.config.model[model]

        if model == "nn":
            trainer = self._setup_nn(model_config)
        elif model == "xgboost":
            # trainer = self._setup_xgboost(model_config)
            pass

        trainer.train()
        trainer.save_model(self.config.model.save_path)

    def _setup_nn(self, model_config):
        train_matches = self.loader.load(model_config.training_path)
        val_matches = self.loader.load(model_config.validation_path)

        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        scaler.fit(train_matches[model_config.features])

        # need loaders
        train_dataset = PLDataset(train_matches, model_config, scaler)
        val_dataset = PLDataset(val_matches, model_config, scaler)

        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=True)

        model = NeuralNet(len(model_config.features), inter_dims=model_config.inter_dims)
        return NNTrainer(model, train_loader, val_loader, scaler, model_config)

    def _get_trainer(self, model_type: str):
        """ method for returning the model-specific trainer """
        if model_type == "nn":
            return self._get_nn_trainer
        elif model_type == "xgboost":
            pass

    def evaluate(self):
        print("evaluating")

    def _get_nn_trainer(self, model_config):
        # what do we need?
        train_matches = self.loader.load(model_config.training_path)
        val_matches = self.loader.load(model_config.validation_path)

        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        scaler.fit(train_matches[model_config.features])

        # need loaders
        train_dataset = PLDataset(train_matches, model_config, scaler)
        val_dataset = PLDataset(val_matches, model_config, scaler)

        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=True)

        model = NeuralNet(len(model_config.features), inter_dims=model_config.inter_dims)
        return NNTrainer(model, train_loader, val_loader, model_config)

        

def main():
    parser = argparse.ArgumentParser(description="Premier League Predictor")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    # Data Pipeline command
    prepare_parser = subparsers.add_parser("prepare", help="Pre-processes raw match data")
    prepare_parser.add_argument("--input_dir", type=str, default="./data/raw", help="Path to folder containing the raw CSV data")
    prepare_parser.add_argument("--save-to", type=str, default="./data/cleaned")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a specific model")
    train_parser.add_argument("--model", type=str, choices=["xgboost", "nn"], required=True)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Runs inference on test set and reports results")
    eval_parser.add_argument("--model", type=str, choices=["xgboost", "nn"], required=True)

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()            
        return
    
    orchestrator = PipelineOrchestrator(args.config)
   
    # need to instantiate a loader and writer
    if args.command == "prepare":
        # Data pipeline logic goes in here 
        # we basically just want to store the cleaned data as one big stacked up DF.
        # we can do the train/val/test split later on based on the information that each season
        orchestrator.process_data()
    elif args.command == "train":
        orchestrator.train(args.model)
    elif args.command == "evaluate":
        orchestrator.evaluate(args.model)


def load_data(self) -> pd.DataFrame:
    pass
    # return raw_df


def test():
    # runs a test example thingamabob
    orc_path = "./config.yaml"
    pl = PipelineOrchestrator(orc_path)
    pl.process_data()


if __name__ == "__main__":
    # BASE_DIR = Path(__file__).resolve().parent
    # CONFIG_PATH = BASE_DIR / "config.yaml"
    main()
    # main()

