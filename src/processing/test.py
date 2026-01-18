from processing.loader import Loader
from processing.writer import Writer
from processing.features import FeatureTransformer
from processing.transform import DataTransformer
from pipeline import DataPipeline
import pandas as pd



def main():
    # test pipeline
    transformer = DataTransformer()
    loader = Loader()
    writer = Writer()

    dp = DataPipeline(loader, transformer, writer)
    dp.run()




if __name__ == "__main__":
    main()