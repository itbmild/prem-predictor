import json
from pathlib import Path
import datetime as dt

class ModelRegistry:
    """
    Local model registry for tracking trained models

    Contains JSON index file storing data for trained models. Each entry
    references a model artifact directory containing all necessary model data
    (weights, scaler, config, metrics)

    Entry is of the form:
    {
      "model_id": str,
      "artifact_path": str,
      "metrics": dict,
      "status": str,
      "time_created": str
      "hyperparameters": dict
    }
    """
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)

        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            # write empty list to registry on instantiation
            self._write_registry([])

        self.registry = self._read_registry()

    def register(self, model_id: str, artifact_path: str, metrics: dict, hyperparams: dict):
        """ Registers a trained model """
        new_entry = {
            "model_id": model_id,
            "artifact_path": artifact_path,
            "metrics": metrics,
            "status": "archived",
            "time_created": dt.now,
            "hyperparameters": hyperparams
        }

        self.registry.append(new_entry)
        self._write_registry(self.registry)
        

    def get_model(self, model_id: str) -> dict:
        """ Returns model referenced by model_id """
        for entry in self.registry:
            if entry["model_id"] == model_id:
                return entry

        raise ValueError(f"Model with id: \'{model_id}\' does not exist")

    def get_latest_model(self) -> dict:
        """ Returns most recently trained model """
        if not self.registry:
            raise ValueError("Registry is empty")
        
        latest = self.registry[0]
        for entry in self.registry[1:]:
            if entry["time_created"] > latest["time_created"]:
                latest = entry
        
        return latest
        
    def get_best(self) -> dict:
        """ Returns entry for current best model """
        for entry in self.registry:
            if entry["status"] == "best":
                return entry
            
        raise ValueError("No best model found")

    def promote(self, model_id: str) -> dict:
        """ 
        Promotes model to best

        Archives existing best model 
        """
        for entry in self.registry:
            if entry["status"] == "best":
                entry["status"] = "archived"

            if entry["model_id"] == model_id:
                entry["status"] = "best"

        self._write_registry(self.registry)

    def archive(self, model_id: str):
        """ Archives model """
        for entry in self.registry:
            if entry["model_id"] == model_id:
                entry["status"] = "archived"

    def _read_registry(self) -> list[dict]:
        """ Read registry JSON file """
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _write_registry(self, registry: list[dict]):
        """ Write registry JSON file """
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)