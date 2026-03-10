""" Class for testing model performance """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from pprint import pprint

from metrics.loss import WDLClassificationMetric

class WDLClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, result):
        
        pass    
    
class Tester:
    def __init__(self, model, test_loader, metric, device):
        self.model = model
        self.test_loader = test_loader
        self.metric = metric
        self.device = device
        # self.scaler = scaler

    def run_inference(self):
        """Takes test data and runs inference through network
         
        Prints accuracy and confusion matrix report
        Plots confusion matrix
        """
        self.model.eval()
        correct = 0
        total_rows = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for (features, labels) in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features)

                # if self.metric is not None:
                count_correct, preds, true = self.metric(outputs, labels)

                correct += count_correct
                total_rows += labels.size(0) # adds current batch length to row acc

                all_labels.extend(true.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            print(f"Accuracy (W/D/L): {100 * correct / total_rows:.2f}%")
            self.cm_report(all_preds, all_labels)
            self.plot_cm(all_preds, all_labels)
    
    def raw_inference(self):
        """ Performs inference with model on given test loader
         
            Returns:
                list[torch.tensor]: List of predicted xG for both home and away: (home_xG, away_xG)
                list[torch.tensor]: List of actual goals scored for home and away: (home_goals, away_goals)
        """
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)                
                outputs = self.model(features)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)


    def cm_report(self, preds, labels):
        """ Prints confusion matrix report given inference results on test data """
        report = classification_report(
            labels,
            preds,
            target_names=["Home Win", "Draw", "Away Win"]
        )
        print(report)

    def plot_cm(self, preds, labels):
        """ plots confusion matrix for WDL classification """
        cm = confusion_matrix(labels, preds)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Home Win", "Draw", "Away Win"]
        )

        fig, ax = plt.subplots(figsize=(10,8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix for W/D/L classification")
        plt.show()

class Evaluator:
    def __init__(self, model: nn.Module, test_loader: DataLoader, device='cpu'):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.metric = WDLClassificationMetric(threshold=0.1)

    def run_inference(self):
        """Takes test data and runs inference through network
         
        Outputs dictionary containing predictions and labels:
        {
          "predictions": list,
          "labels": list    
        }
        """
        self.model.eval()
        correct = 0
        total_rows = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for (features, labels) in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features)

                # if self.metric is not None:
                count_correct, preds, true = self.metric(outputs, labels)

                correct += count_correct
                total_rows += labels.size(0) # adds current batch length to row acc

                all_labels.extend(true.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        return {
            "predictions": all_preds,
            "labels": all_labels,
            "correct": correct,
            "rows": total_rows
        }

            # print(f"Accuracy (W/D/L): {100 * correct / total_rows:.2f}%")
            # self.cm_report(all_preds, all_labels)
            # self.plot_cm(all_preds, all_labels)

        # by this point we have reached the end of the inference, we have all of our predictions and our labels

    
    def raw_inference(self):
        """ Performs inference with model on given test loader
         
            Returns:
                list[torch.tensor]: List of predicted xG for both home and away: (home_xG, away_xG)
                list[torch.tensor]: List of actual goals scored for home and away: (home_goals, away_goals)
        """
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)                
                outputs = self.model(features)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)
    
    def _generate_report(self, preds, labels):
        """ Prints confusion matrix report given inference results on test data """
        report = classification_report(
            labels,
            preds,
            target_names=["Home Win", "Draw", "Away Win"],
            output_dict=True
        )
        return report
    
    def _generate_confusion_matrix(self, preds, labels) -> Figure:
        """ plots confusion matrix for WDL classification """
        cm = confusion_matrix(labels, preds)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Home Win", "Draw", "Away Win"]
        )

        # fig, ax = plt.subplots(figsize=(10,8))
        disp.plot(cmap=plt.cm.Blues)
        fig = disp.ax_.figure
        return fig


        # plt.title("Confusion Matrix for W/D/L classification")
        # plt.show()

    def _generate_figures(self, predictions: list, labels: list) -> list[Figure]:
        """ Takes predictions and ground truth labels and returns list of figures """
        figures = []

        confusion_matrix = self._generate_confusion_matrix(predictions, labels)
        figures.append(confusion_matrix)

        # TODO add more figures here, could be useful to have ROC curves or other shit

        return figures



    def evaluate(self) -> dict:
        """ Called by the orchestrator to run full evaluation logic
         
        Handles inference loop on test set and stores report
        """

        eval_output = self.run_inference()
        
        predictions = eval_output["predictions"]
        labels = eval_output["labels"]

        report = self._generate_report(predictions, labels)
        figures = self._generate_figures(predictions, labels)


        confusion_matrix = self._generate_confusion_matrix(predictions, labels)

        # eval = self._get_eval(report, figures)

        eval = {
            "report": report,
            "figures": {
                "cm": confusion_matrix
            }
        }


        return eval