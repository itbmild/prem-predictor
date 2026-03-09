""" CLI, entrypoint for Pipeline """
import argparse
import yaml
import pandas as pd
import torch
from datetime import datetime as dt
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

# model saving (WIP)
from registry.registry import ModelRegistry
from registry.model_saver import ModelSaver
from registry.model_loader import ModelLoader

class PipelineOrchestrator:
    """ Orchestrator class for data processing / model training / model evaluation """
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = Config(yaml.safe_load(f))

        self.loader = Loader()
        self.writer = Writer()

        # model saving classes
        self.registry = ModelRegistry(self.config.registry_path)
        self.saver = ModelSaver(self.config.artifacts_path)
        self.model_loader = ModelLoader(self.registry)


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
        model_id = self._generate_model_id()
        metrics = trainer.get_metrics()

        artifact_path = self.saver.save_model(
            model_id=model_id,
            model=trainer.get_model(),
            scaler=trainer.get_scaler(),
            config=model_config,
            metrics=metrics
        )

        self.registry.register(
            model_id=model_id,
            artifact_path=artifact_path,
            metrics=metrics,
            hyperparams=trainer.get_hyperparams()
        )
        
        # lets check to see if our model has been stored with the correct hyperparameters
        # what is needed for this? our reporting mechanism should belong to the registry class
        # we should have a function in our registry that prints out valuable information about each model


        # lists all of the trained models and their information
        self.registry.list_models()

        # describes the model with the given id, basically just prints the corresponding entry
        self.registry.describe_model(model_id)


        # trainer.save_model(self.config.model.save_path)

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

    def _generate_model_id(self) -> str:
        """ Generates unique identifier for trained model """
        time = dt.now()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # return f"model_{timestamp}"
        return timestamp
        

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

