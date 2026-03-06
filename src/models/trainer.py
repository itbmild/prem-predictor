import torch
from models.modules import NeuralNet
from dataset import PLDataModule

class NNTrainer:
    """ Trainer class for Neural Network """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []

    def __init__(self, train_loader, val_loader, config):
        self.config = config
        # Hyperparameters
        self.epochs = config.epochs
        self.lr = config.lr
        input_dims = len(config.features)
        output_dims = len(config.labels)
        self.model = NeuralNet(
            input_dims=input_dims,
            inter_dims=self.config.inter_dims,
            output_dims=output_dims
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _run_epoch(self, loader, training: bool):
        """ Private method for one pass """
        if training:
            self.model.train()
        else:
            self.model.eval()

        loss_acc = 0.0
        for features, labels in loader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            # forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_acc += loss.item()
        return loss_acc / len(loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.5f} Validation Loss: {val_loss:.5f}")    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def get_model(self):
        return self.model
    

class XGBTrainer:
    """ Trainer for XGBoost model """
    def __init__(self, model):
        pass

    