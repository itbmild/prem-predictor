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
    

       