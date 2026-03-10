from torch.utils.data import Dataset, DataLoader
import torch
# from ..models.constants import PREM_FEATURES, PREM_LABELS, PREM_EVAL_LABELS, PREM_COLS_TO_DROP
from file_io.loader import Loader
from sklearn.preprocessing import StandardScaler
import pandas as pd
 
class PLDataset(Dataset):
    """
    Stores match data for each premier league season
    """
    def __init__(self, matches: pd.DataFrame, config, scaler=None, eval=False):
        self.features = matches[config.features]

        if scaler is not None:
            self.features = scaler.transform(self.features)

        self.labels = matches[config.eval_labels if eval else config.labels]

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx): 
        """ need to return the aggregated data as well as the score """
        # needs to return in the form of the data as well as the labels
        X = self.features.iloc[idx]
        Y = self.labels.iloc[idx]

        X_in, Y_out = torch.tensor(X.values, dtype=torch.float32), torch.tensor(Y.values, dtype=torch.float32)
        return (X_in, Y_out)
    
class PLDataModule:
    """ Class for managing dataloaders to NN model """
    def __init__(self, model_config):
        # self.train_path = train_path
        # self.val_path = val_path
        # self.test_path = test_path
        # self.batch_size = batch_size
        # self.scaler = StandardScaler().set_output(transform="pandas")
        # if scale:
        #     self.fit_scaler()
        self.cfg = model_config
        self.train_path = model_config.training_path
        self.val_path = model_config.validation_path
        self.test_path = model_config.test_path

        self.batch_size = model_config.batch_size
        if model_config.scale:
            self.scaler = StandardScaler().set_output(transform="pandas")
            self.fit_scaler()


    # def fit_scaler(self):
    #     """ Fit scaler to the training data """
    #     loader = Loader()
    #     train_df = loader.load(self.train_path)

    #     # need to change this 
    #     # train_features = train_df.drop(columns=PREM_COLS_TO_DROP)
    #     # to this
    #     train_features = train_df[self.cfg.features]
    #     print(len(train_features))
    #     self.scaler.fit(train_features)

    # def get_train_loader(self):
    #     """ Returns DataLoader object for training data """
    #     dataset = PremierLeagueDataset(self.train_path, scaler=self.scaler)
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # def get_val_loader(self):
    #     """ Returns DataLoader object for validation data """
    #     dataset = PremierLeagueDataset(self.val_path, scaler=self.scaler)
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    # def get_test_loader(self):
    #     """ Returns DataLoader object for test data """
    #     dataset = PremierLeagueDataset(self.test_path, scaler=self.scaler, eval=True)
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=False) 
    
    # def get_scaler(self):
    #     return self.scaler
    