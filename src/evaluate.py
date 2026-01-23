""" Entrypoint for running inference on test data """
from models.tester import Tester
from models.dataset import PLDataModule
from models.loss import WDLClassificationMetric
from models.modules import NeuralNet
import torch

from utils import TEST_DATA_PATH, MODEL_PATH

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_manager = PLDataModule(test_path=TEST_DATA_PATH)
    test_loader = loader_manager.get_test_loader()

    metric = WDLClassificationMetric(threshold=0.2)
    model = NeuralNet(18,64,2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    tester = Tester(model, test_loader, metric, device)
    tester.run_inference()

if __name__ == "__main__":
    main()