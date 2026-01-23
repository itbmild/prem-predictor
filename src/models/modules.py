""" Neural Network modules for 2 layer perceptron """
import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        return self.block(x)
    

class NeuralNet(nn.Module):
    def __init__(self, input_dims, inter_dims, output_dims=2):
        super(NeuralNet, self).__init__()
        
        self.block1 = BasicBlock(input_dims, inter_dims)
        self.block2 = BasicBlock(inter_dims, inter_dims)
        self.block3 = BasicBlock(inter_dims, inter_dims)
        self.regression = nn.Linear(inter_dims , output_dims)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.regression(x)
        x = torch.nn.functional.softplus(x)
        return x
    