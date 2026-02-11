""" Entrypoint for running inference on test data """
from models.tester import Tester
from models.dataset import PLDataModule
from models.loss import WDLClassificationMetric
from models.modules import NeuralNet
from processing.writer import Writer
from processing.loader import Loader
import pandas as pd

import torch

from utils import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, PREDICTIONS_PATH, PREDICTION_COLS

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_manager = PLDataModule(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH)
    test_loader = loader_manager.get_test_loader()

    metric = WDLClassificationMetric(threshold=0.1)

    model = NeuralNet(29,64,2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    tester = Tester(model, test_loader, metric, device)
    # tester = prepare_model()


    # tester.run_inference()

    preds, labels = tester.raw_inference() # returns predictions and labels as numpy arrays
    col_names = PREDICTION_COLS
    preds_df = pd.DataFrame(preds, columns=col_names)



    writer = Writer()

    writer.save_to_dir(preds_df, PREDICTIONS_PATH, "test_outputs.csv")
    # we have the predicted xG values for each of the games, and the actual goals scored
    # we want to perform our sampling method
    # how should we do this?
    
    # print(labels)
    # print(len(labels))

def prepare_model() -> Tester:
    """ Loads model to be evaluated, returns Tester object """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_manager = PLDataModule(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, scale=True)

    test_loader = loader_manager.get_test_loader()
    scaler = loader_manager.get_scaler()

    model = NeuralNet(29,64,2)
    tester = Tester(model, test_loader, None, device, scaler)
    return tester


def get_xG_scores():
    """ returns predicted goal scores from model output """
    loader = Loader()
    predicted_scores = loader.load(PREDICTIONS_PATH)
    return predicted_scores

if __name__ == "__main__":
    main()