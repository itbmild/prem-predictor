from pathlib import Path
import torch
import joblib
import yaml
import json

class ModelSaver:
    """ 
    Class for saving model artifacts locally
    """
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)

    def save_model(self, model_id, model, scaler, config, metrics) -> Path:
        """ Saves model artifact to artifacts directory and returns path """
        artifact_path = self.artifacts_dir / model_id
        artifact_path.mkdir(parents=True, exist_ok=False)

        # save model weights
        torch.save(model.state_dict(), artifact_path / "model.pth")

        # save scaler alongside model
        joblib.dump(scaler, artifact_path / "scaler.pkl")

        # save config used for model architecture / hyperparameters
        with open(artifact_path / "config.yaml", "w") as f:
            yaml.dump(dict(config), f)

        # save metrics so model can be compared to other models
        with open(artifact_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return str(artifact_path)
        

    