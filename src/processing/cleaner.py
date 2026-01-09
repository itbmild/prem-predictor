""" Class for cleaning dataset (creates season table and removes unwanted cols from data) """
import pandas as pd

class DataCleaner:
    def __init__(self, target_cols: list):
        self.target_cols = target_cols

    def clean(self, df: pd.DataFrame):
        # here is where we keep certain columns
        df = df[self.target_cols]

        pass