""" Writer class to handle saving files """
import pandas as pd
from pathlib import Path

class Writer:
    def save_to_dir(self, df: pd.DataFrame, dir: str, filename: str):
        """Saves given pandas dataframe to directory in CSV format

        Args:
            df (pd.DataFrame): dataframe object to be saved
            dir (str): directory to save DataFrame to
            filename (str): name given to saved file
        """
        target = Path(dir) / f"{filename}.csv"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
        except Exception as e:
            print(f"could not save to specified directory with exception: {e}")

    def batch_save_to_dir(self, dfs: list[pd.DataFrame], dir: str, start: int, filename: str):
        """Saves list of pandas dataframe objects in CSV format
        
        Args:
            dfs (list[pd.DataFrame]): list of DataFrames to be saved
            dir (str): directory to which DataFrames are saved
            start (int): starting integer, used for ordering and identifying saved files
            filename (str): name given to files preceding current index
        """
        year = start
        for df in dfs:
            self.save_to_dir(df, dir, f"{filename}-{year}")
            year += 1
