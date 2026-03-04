from torch.utils.data import Dataset, DataLoader
import torch
from .constants import PREM_FEATURES, PREM_LABELS, PREM_EVAL_LABELS, PREM_COLS_TO_DROP
from processing.loader import Loader
from sklearn.preprocessing import StandardScaler
import pandas as pd
 
class PremierLeagueDataset(Dataset):
    """
    Stores match data for each premier league season
    """
    def __init__(self, match_path, scaler=None, eval=False):
        self.loader = Loader()
        self.match_path = match_path
        self.matches = self.loader.load(match_path)
        self.features = self.matches.drop(columns=PREM_COLS_TO_DROP)
        print(self.features.columns)
        self.scaler = scaler

        if scaler is not None:
            self.features = scaler.transform(self.features)

        if eval:
            self.labels = self.matches[PREM_EVAL_LABELS]
        else:
            self.labels = self.matches[PREM_LABELS]

    def __len__(self):
        return len(self.matches)
    
    def __getitem__(self, idx): 
        """ need to return the aggregated data as well as the score """
        # needs to return in the form of the data as well as the labels
        X = self.features.iloc[idx]
        Y = self.labels.iloc[idx]

        X_in, Y_out = torch.tensor(X.values, dtype=torch.float32), torch.tensor(Y.values, dtype=torch.float32)
        return (X_in, Y_out)
    
class PLDataModule:
    """ Class for managing dataloaders to NN model """
    def __init__(self, train_path="", val_path="", test_path="", batch_size=64, scale=True):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.scaler = StandardScaler().set_output(transform="pandas")
        if scale:
            self.fit_scaler()

    def fit_scaler(self):
        """ Fit scaler to the training data """
        loader = Loader()
        train_df = loader.load(self.train_path)
        train_features = train_df.drop(columns=PREM_COLS_TO_DROP)
        self.scaler.fit(train_features)

    def get_train_loader(self):
        """ Returns DataLoader object for training data """
        dataset = PremierLeagueDataset(self.train_path, scaler=self.scaler)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        """ Returns DataLoader object for validation data """
        dataset = PremierLeagueDataset(self.val_path, scaler=self.scaler)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self):
        """ Returns DataLoader object for test data """
        dataset = PremierLeagueDataset(self.test_path, scaler=self.scaler, eval=True)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False) 
    
    def get_scaler(self):
        return self.scaler