""" Class for loading data from source and returning pd dataframe """
import pandas as pd
from pathlib import Path

class Loader:
    """
    Loads Premier league season as dataframe given filepath
    """

    def load(self, file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            raise Exception(f"File not found: {e}")

    def load_batch(self, filepaths: list[str]) -> list[pd.DataFrame]:
        seasons = []
        for path in filepaths:
            raw_season = self.load(path)
            seasons.append(raw_season)
        return seasons
    
    def load_batch_concat(self, filepaths: list[str]) -> list[pd.DataFrame]:
        """ Takes list of filepaths and returns concatenated dataframe """
        df = pd.DataFrame()
        for path in filepaths:
            season = self.load(path)
            pd.concat([df, season])
        return df

    def get_files(self, filepath: str) -> list:
        """ returns list of filepath for CSV files """
        # need to open the directory, return the files as a list
        return list(Path(filepath).glob("*.csv"))
    
        