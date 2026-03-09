import json
from pathlib import Path
from datetime import datetime as dt

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
            "time_created": dt.now().strftime("%Y/%m/%d %H:%M:%S"),
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

    def list_models(self):
        """ Prints information about each model in the registry to stdout """
        if not len(self.registry):
            # registry is empty
            print("no models have been registered, to register a model use train argument")
            return
        
        bar = "-" * 80
        print("\nTRAINED MODELS")
        print(bar)
        print(f"{'MODEL ID':20} {'STATUS':10} {'VAL ACC':10} {'VAL LOSS':10} {'TIME CREATED':10}")
        print(bar)

        # iterate through registry
        for entry in self.registry:
            model_id = entry["model_id"]
            status = entry["status"]

            metrics = entry["metrics"]
            val_acc = metrics["acc"]
            val_loss = metrics["loss"]

            time_created = entry["time_created"]

            print(f"{model_id:20} {status:10} {val_acc:<10.4f} {val_loss:<10.4f} {time_created:10}")

    def describe_model(self, model_id: str):
        # NOTE if we decide to add more model types, we need 
        # to either do model specific printing or generic dict printing for sections
        """ Prints all information stored in registry about a specific model to stdout """
        entry = self.get_model(model_id)
        
        bar = "-" * 60
        print("\nMODEL DETAILS")
        print(bar)

        print(f"Model ID:         {entry["model_id"]}")
        print(f"Status:           {entry["status"]}")
        print(f"Time Created:     {entry["time_created"]}")
        print(f"Artifact Path:    {entry["artifact_path"]}")

        print("\nMETRICS")
        print(bar)

        metrics = entry["metrics"]
        print(f"val accuracy:     {metrics["acc"]:.4f}")
        print(f"val loss:         {metrics["loss"]:.4f}")

        print("\nHYPERPARAMETERS")
        print(bar)

        hyperparams = entry["hyperparameters"]
        print(f"learning rate: {hyperparams["learning rate"]}")
        print(f"epochs:        {hyperparams["epochs"]}")
        print(f"weight decay:  {hyperparams["weight decay"]}")  
        print(f"batch size:    {hyperparams["batch size"]}")


