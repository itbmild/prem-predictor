""" Class for loading data from source and returning pd dataframe """
import pandas as pd

class PremSeasonLoader:
    """
    Loads Premier league season as dataframe given filepath
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError as e:
            raise Exception(f"File not found: {e}")
