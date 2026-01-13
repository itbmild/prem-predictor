from torch.utils.data import Dataset, DataLoader

class PremierLeagueDataset(Dataset):
    """
    Stores match data for each premier league season
    """
    def __init__(self, match_dir):
        self.match_data = self.load_match_data(PROCESSED_MATCH_DATA_DIR)

    def load_match_data(self):
        pass
    
    def __len__(self):
        return len(self.match_data)
    
    def __getitem__(self, idx):
        """ need to return the aggregated data as well as the score """
        pass