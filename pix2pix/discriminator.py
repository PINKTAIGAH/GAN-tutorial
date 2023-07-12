import torch
import torch.nn as nn


"""
Class containing a single convolutional block for the neural networks
"""
class CNNBlock(nn.Module):
    def __init__(self, inClannels, outChannels, stride=2):
        super().__init__()
        
        self.conv = nn.Sequential(
            ### Padding mode is set to false to avoid artefacts
            nn.Conv2d(inClannels, outChannels, kernel_size=4,
                      stride=stride, bias=False, padding_mode=False)
            nn.BatchNorm2d(outChannels)
            nn.LeakyReLU(0.2)
        )
