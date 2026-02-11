""" Class for testing model performance """
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

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

