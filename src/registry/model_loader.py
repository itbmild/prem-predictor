from registry.registry import ModelRegistry
from pathlib import Path
from models.modules import NeuralNet
from config import Config
from sklearn.preprocessing import StandardScaler
import yaml
import torch
import joblib

""" Class for loading model given artifact path from model_id """
class ModelLoader:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def load_artifact(self, model_id: str) -> dict:
        """ loads saved model artifact given model_id

        Artifact identified by ID and path is resolved through registry
        """
        entry = self.registry.get_model(model_id) # registry entry, we need artifact path

        if entry is None:
            raise ValueError(f"Model with ID: '{model_id}' does not exist")

        artifact_path = Path(entry["artifact_path"])

        model_config = self._load_config(artifact_path)
        model = self._load_model(artifact_path, model_config)
        scaler = self._load_scaler(artifact_path)
        return {
            "model": model,
            "model_config": model_config,
            "scaler": scaler
        }

    def _load_config(self, artifact_path: Path) -> dict:
        """ Takes path to model artifact and returns config for that model """
        with open(artifact_path / "config.yaml", "r") as f:
            return Config(yaml.safe_load(f))
        
    def _load_model(self, artifact_path: str, config: Config):
        """ Loads model given path to artifact directory containing it """
        model = NeuralNet(
            input_dims=len(config.features),
            inter_dims=config.inter_dims
        )
        state_dict = torch.load(artifact_path / "model.pth")
        model.load_state_dict(state_dict)
        return model
    
    def _load_scaler(self, artifact_path: Path) -> StandardScaler:
        """ Loads scaler given path to model artifact, returns as resolved scaler object """
        return joblib.load(artifact_path / "scaler.pkl")