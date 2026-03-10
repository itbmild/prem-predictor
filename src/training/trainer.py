import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import joblib
from models.modules import NeuralNet
# from dataset import PLDataModule
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from pathlib import Path

from config import Config

class NNTrainer:
    """ Trainer class for Neural Network """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, scaler: StandardScaler, config: Config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler
        self.config = config
        # Hyperparameters
        self.epochs = config.epochs
        self.lr = config.lr
        self.wd = config.weight_decay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)

        self.train_losses = []
        self.val_losses = []

    def _run_epoch(self, loader: DataLoader, training: bool):
        """ Private method for one pass """
        if training:
            self.model.train()
        else:
            self.model.eval()

        loss_acc = 0.0
        for features, labels in loader:
            # features, labels = features.to(self.device), labels.to(self.device)
            
            # forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_acc += loss.item()
        return loss_acc / len(loader)

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {train_loss:.5f} Validation Loss: {val_loss:.5f}")    

    def save_model(self, path):
        """ Saves model with scaler and config file """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(path) / "neural_networks" / f"nn_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, save_dir / "scaler.pkl")

        with open(save_dir / "model_config.yaml", "w") as f:
            yaml.dump(dict(self.config), f)
        
        torch.save(self.model.state_dict(), save_dir / "model.pth")

    def get_model(self):
        return self.model

    def get_scaler(self):
        return self.scaler
    
    def get_metrics(self):
        """ Returns accuracy and loss metrics for performance on validation set of trained model """
        return {
            "acc": 0,
            "loss": self.val_losses[-1]
        }
    
    def get_hyperparams(self):
        """ Returns dict containing hyperparameter information
         
        Used for comparing trained models in registry  
        """
        return {
            "learning rate": self.lr,
            "epochs": self.epochs,
            "weight decay": self.wd,
            "batch size": self.config.batch_size
        }
    
    def get_architecture(self):
        """ Returns dict containing architecture information about the network """
        return {
            "input dims": len(self.config.features)
        }