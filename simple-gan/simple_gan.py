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
    def __init__(self, imageDimention):        # input features size will be 784
        super().__init__()
        self.discriminaor = nn.Sequential(
            nn.Linear(imageDimention, 128),
            nn.LeakyReLU(0.1),              # 0.1 is an arbitrary hyperparam
            nn.Linear(128, 1),              
            # This network is binary so 1 output --> Real or fake

            nn.Sigmoid(),                   # So we get probabilities as output
        )

    def forward(self, x):
        return self.discriminaor(x)

"""
We now look to define our Generator that will be tasked with generating 
imitations of the input data. This will take random nouse and use it to generate
an image of the same dimentions and the dataset we are using
"""

class Generator(nn.Module):
    def __init__(self, zDimention, imageDimention):
        # zDimiention is the dimention of the latent noise the generator uses
        # to make randomized datasets

        super().__init__()

        self.generator = nn.Sequential(
            nn.Linear(zDimention, 256),
            nn.LeakyReLU(0.1),
            nn.LeakyReLU(256, imageDimention),
            nn.Tanh(),                  # make output pixel val between -1 and 1
        )

        def forward(self, x):
            return self.generator(x)
