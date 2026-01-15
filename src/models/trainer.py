import torch

class Trainer:
    """ Class for training premier league prediction model """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

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

            loss_acc += loss.item() * features.size(0)
        return loss_acc / len(loader.dataset)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)