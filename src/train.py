""" Entrypoint for model training """
from models.trainer import Trainer
from torch.utils.data import DataLoader
from models.dataset import PremierLeagueDataset, PLDataModule
import torch

from utils import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH

def main():
    # select device so pytorch functions can be parallelised if desired
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 
    loader_manager = PLDataModule(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH)
    train_loader = loader_manager.get_train_loader()
    val_loader = loader_manager.get_val_loader()
    print(train_loader)
    print(val_loader)

if __name__ == "__main__":
    main()