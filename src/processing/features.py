import pandas as pd
from abc import ABC, abstractmethod

class BaseFeatures(ABC):
    def prepare(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class RollingWindowFeatures(BaseFeatures):
    def generate(self, df: pd.DataFrame):
        pass

class HeadToHeadFeatures(BaseFeatures):
    def generate(self, df: pd.DataFrame):
        pass

class RollingWindowFeatures(BaseFeatures):
    def prepare(self, prev_season: pd.DataFrame):
        self.prev_season = prev_season
    
    def generate(self, df: pd.DataFrame):
        pass