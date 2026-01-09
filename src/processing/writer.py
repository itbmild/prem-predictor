""" Writer class to handle saving files """
import pandas as pd
from pathlib import Path
from loader import Loader


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


if __name__ == "__main__":
    loader = Loader('./data/raw')
    files = loader.get_files()
    test = loader.load(files[0])


    writer = Writer()
    writer.save_to_dir(test, './data/processed/processed_yearly', 'prem-season-2015.csv')

