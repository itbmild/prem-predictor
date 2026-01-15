""" Writer class to handle saving files """
import pandas as pd
from pathlib import Path
from .loader import Loader

class Writer:
    def __init__(self):
        pass

    def save_to_dir(self, df: pd.DataFrame, dir: str, filename: str):
        target = Path(dir) / filename
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
        except Exception as e:
            print(f"could not save to specified directory with exception: {e}")

    def batch_save_to_dir(self, dfs: list[pd.DataFrame], dir: str, starting_year: int):
        year = starting_year
        for df in dfs:
            self.save_to_dir(df, dir, f"processed-season-{year}")
            year += 1
