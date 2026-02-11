""" Class for loss functions """
import torch.nn as nn
import torch
import pandas as pd

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

    def forward(self, pred: torch.tensor, results: torch.tensor):
        """ Takes predicted xG and actual goals scored in the match
        Computes W/D/L based on predicted xG 

        Args:
            pred: predicted xG per match, 

        Returns:
            torch.tensor: wdl encoding given by the predicted xG for the home team
            torch.tensor: correct 

        """
        # print(f"shape of pred:{pred.shape}")
        # print(f"type of pred: {type(pred)}")
        # print(f"type of result: {type(results)}")
        diff = pred[:,0] - pred[:, 1] # score difference (home xG - away xG)
        wdl_encoding = torch.zeros(diff.shape[0], device=pred.device) 

        # encode each match row based on result
        wdl_encoding[diff > self.threshold] = 0
        wdl_encoding[torch.abs(diff) <= self.threshold] = 1
        wdl_encoding[diff < -self.threshold] = 2

        # get correct wdl from labels, will be column containing 1 in (W,D,L)
        correct_result = results.argmax(dim=1)
        count_correct = (wdl_encoding == correct_result).sum().item()
        
        return count_correct, wdl_encoding, correct_result
    
    # def forward(self, preds: torch.tensor) -> torch.tensor:
    #     """ performs forward pass given batch of predictions and corresponding labels

    #         Args:
    #             preds (torch.tensor): tensor containing predicted xG, shape: (batch_size, 3)
            
    #         Returns:
    #             torch.tensor: predicted xG for both the home and away team, shape (batch_size, 2)
    #     """
    #     # starting to think i dont need a critera at all then?


    def report(self):
        cm = self.matrix.cpu().numpy()
        names = ["Home Win", "Draw", "Away Win"]

class BrierScore(nn.Module):
    """Performance metric for results of Monte-Carlo simulation

    Measures mean squared difference between predicted probability assigned to possible
    outcomes for a given "row" or match, and the actual outcome of that match     
    
    BS = 1/N sum 1->N (ft - ot)^2

    Where ft is forecasted probability and ot is outcome of event at instance t (0 if it doesnt happen, one if it does)
    N is number of rows

    ^ above is only for single category

    Multi category Brier Score is given by:

    BS = 1/N sum 1->N sum 1->R (fti - oti)^2

    R is the number of possible classes in which the event can fall, and N is the overall number of instances of all classes.
    fti is predicted probability of class i
    oti is 1 if it is the ith class instance t; 0 otherwise. 

    in our case, R = 3

    We have all of our predictions already as a dataframe, do we need a separate class for the brier? probably not


    ""


    pass



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
    
       