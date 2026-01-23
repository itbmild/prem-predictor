""" Class for loss functions """
import torch.nn as nn
import torch

class JointPoissonLoss(nn.Module):
    def __init__(self):
        super(JointPoissonLoss, self).__init__()
        self.activation = nn.Softplus()

    def forward(self, preds, targets): 
        # compute loss independently for each dim
        loss = self.activation(preds) - targets * preds 
        # sum each lambda in each observation
        per_sample = loss.sum(dim=1)
        # return mean across batches
        return per_sample.mean()

 
class WDLClassificationMetric(nn.Module):
    """ 
    Computes the loss for a specific match prediction based on xG 
    Cross Entropy Loss between the predicted outcome based on xG and
    The true W/D/L outcome
    """
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, results):
        diff = pred[:,0] - pred[:, 1]
        wdl_encoding = torch.zeros(diff.shape[0], device=pred.device)

        wdl_encoding[diff > self.threshold] = 0
        wdl_encoding[torch.abs(diff) <= self.threshold] = 1
        wdl_encoding[diff < -self.threshold] = 2

        # get correct wdl from labels
        correct_result = results.argmax(dim=1)
        count_correct = (wdl_encoding == correct_result).sum().item()
        
        return count_correct, wdl_encoding, correct_result
    
    def report(self):
        cm = self.matrix.cpu().numpy()
        names = ["Home Win", "Draw", "Away Win"]

        
if __name__ == "__main__":
    metric = WDLClassificationMetric(threshold=0.1)

    test_predictions = torch.tensor([
        [2.0, 1.0],
        [1.05, 1.0],
        [0.5, 2.0]
    ])

    test_results = torch.tensor([
        [1,0,0],
        [0,1,0],
        [1,0,0]
    ])

    count = metric(test_predictions, test_results)

    print(f"Total Matches: {len(test_results)}")
    print(f"Correct Games: {count}")
    print(f"Accuracy: {count/len(test_results):.2%}")
    
       