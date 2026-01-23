""" Class for testing model performance """
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

    def run_inference(self):
        self.model.eval()
        
        correct = 0
        total_rows = 0
        
        all_preds = []
        all_labels = []

        flag = True
        print(self.test_loader)
        with torch.no_grad():
            for (features, labels) in self.test_loader:
                # if flag:
                #     # print("example feature input: ")
                #     # print(features[0])
                #     # print("example corresponding WDL label:")
                #     # print(labels[0])
                #     flag = False


                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features)
                count_correct, preds, true = self.metric(outputs, labels)

                correct += count_correct
                total_rows += labels.size(0)

                all_labels.extend(true.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            print(f"Accuracy (W/D/L): {100 * correct / total_rows:.2f}%")
            self.cm_report(all_preds, all_labels)
            self.plot_cm(all_preds, all_labels)
    
    def cm_report(self, preds, labels):
        report = classification_report(
            labels,
            preds,
            target_names=["Home Win", "Draw", "Away Win"]
        )
        print(report)

    def plot_cm(self, preds, labels):
        print("reached")
        cm = confusion_matrix(labels, preds)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Home Win", "Draw", "Away Win"]
        )

        
        fig, ax = plt.subplots(figsize=(10,8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix for W/D/L classification")
        plt.show()

