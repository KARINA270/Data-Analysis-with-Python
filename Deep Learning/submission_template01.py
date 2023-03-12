import numpy as np
import torch
from torch import nn

def create_model():
    class Net(nn.Module):
        def __init__(self, dim):
            dim = 784
            super(Net, self).__init__()
            self.L1 = nn.Linear(784, 256, bias = True)
            self.L2 = nn.Linear(256, 16, bias = True)
            self.L3 = nn.Linear(16, 10, bias = True)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.L1(x)
            x = self.activation(x)
            x = self.L2(x)
            x = self.activation(x)
            x = self.L3(x)
            return self.activation(x)

def count_parameters(model):
    medium_model = nn.Sequential(nn.Linear(784, 256, bias=True), nn.ReLU(), nn.Linear(256, 16, bias=True),nn.Linear(16, 10, bias=True),nn.ReLU())
    assert count_parameters(medium_model) == ((784 * 256+256) *16+16) * 10
    return(count_parameters(medium_model))
   
