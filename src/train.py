""" Entrypoint for model training """
from models.trainer import Trainer
from torch.utils.data import DataLoader
from models.dataset import PremierLeagueDataset, PLDataModule
from models.modules import NeuralNet
from models.loss import JointPoissonLoss
import torch

from utils import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, MODEL_PATH

def main():
    # select device so pytorch functions can be parallelised if desired
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 
    loader_manager = PLDataModule(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH)
    train_loader = loader_manager.get_train_loader()
    val_loader = loader_manager.get_val_loader()

    # criterion = JointPoissonLoss()
    criterion = torch.nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean')
    learning_rate = 0.0005
    decay=1e-2

    # need in dims, inter dims and out dims
    model = NeuralNet(4,64,2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    num_epochs=100
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(num_epochs)
    trainer.save_model(MODEL_PATH)

if __name__ == "__main__":
    main()