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
        self.train_losses = []
        self.val_losses = []

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

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.5f} Validation Loss: {val_loss:.5f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)