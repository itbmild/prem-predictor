from torch.utils.data import Dataset, DataLoader
import torch
from utils import PROCESSED_MATCH_DATA_DIR, PREM_FEATURES, PREM_LABELS
from src.processing.loader import Loader
import pandas as pd


class PremierLeagueDataset(Dataset):
    """
    Stores match data for each premier league season
    """
    def __init__(self, match_path):
        self.loader = Loader()
        self.match_path = match_path
        self.matches = self.loader.load(match_path)
        self.features = self.matches[PREM_FEATURES]
        self.labels = self.matches[PREM_LABELS]

    def __len__(self):
        return len(self.matches)
    
    def __getitem__(self, idx): 
        """ need to return the aggregated data as well as the score """
        # needs to return in the form of the data as well as the labels
        X = self.features.iloc[idx]
        Y = self.labels.iloc[idx]

        X_in, Y_out = torch.tensor(X), torch.tensor(Y)
        return (X_in, Y_out)
    
class PLDataModule:
    """ Class for managing dataloaders to NN model """
    def __init__(self, train_path, val_path, test_path, batch_size=20):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size

    def get_train_loader(self):
        dataset = PremierLeagueDataset(self.train_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        dataset = PremierLeagueDataset(self.val_path)
        return DataLoader(dataset, batch_size=20, shuffle=False)

    def get_test_loader(self):
        dataset = PremierLeagueDataset(self.test_path)
        return DataLoader(dataset, batch_size=20, shuffle=False)     