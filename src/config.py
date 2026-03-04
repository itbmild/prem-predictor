""" Class for handling config dictionary loaded from yaml """
from pathlib import Path

class Config(dict):
    """ 
    Wrapper for config dict, provides methods for getting absolute paths
    and dot notation
    """
    def __getattr__(self, key: str):
        value = self[key]
        if isinstance(value, dict):
            return Config(value)
        else:
            return value
    
    def path(self, path: str) -> str:
        BASE = Path(__file__).resolve().parent.parent
        abs_path = BASE / self[path]
        return abs_path