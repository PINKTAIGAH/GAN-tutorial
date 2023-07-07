"""
In this model we will use the MNIST data to 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

"""
We first define the class for our Discriminaor. This usially takes the form of
what a regular NN model for the tak we want to be. In this case it is a simple
fully connected feedforward network
"""

class Discriminaor(nn.Module):
    def __init__(self, in_features):        # input features size will be 784
        super().__init__()
        self.discriminaor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),              # 0.1 is an arbitrary hyperparam
            nn.Linear(128, 1),              
            # This network is binary so 1 output --> Real or fake

            nn.Sigmoid(),                   # So we get probabilities as output
        )

    def forward(self, x):
        return self.discriminaor(x)

"""
We now look to define our Generator that will be tasked with generating 
imitations of the input data
"""

