""" Class for loading data from source and returning pd dataframe """
import pandas as pd
from pathlib import Path

class Loader:
    """
    Loads Premier league season as dataframe given filepath
    """
    def __init__(self, directory_path: str):
        self.dir_path = directory_path

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

    def get_files(self) -> list:
        """ returns list of filepath for CSV files """
        # need to open the directory, return the files as a list
        return list(Path(self.dir_path).glob("*.csv"))
    
        