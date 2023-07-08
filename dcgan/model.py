import torch
import torch.nn as nn
from torch.nn.modules import padding

"""
Discriminator model
"""

class Discriminator(nn.Module):

    def __init__(self, channelImages, featuresD):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            ### INPUT SIZE: N * channelImages * 64 * 64
            nn.Conv2d(channelImages, featuresD, kernel_size=4,
                      stride=2, padding= 1),
            ### SIZE: 32*32
            nn.LeakyReLU(0.2),
            self._block(featuresD, featuresD*2, 4, 2, 1),
            ### SIZE: 16*16
            self._block(featuresD*2, featuresD*4, 4, 2, 1),
            ### SIZE: 8*8
            self._block(featuresD*4, featuresD*8, 4, 2, 1),
            ### SIZE: 4*4
            nn.Conv2d(featuresD*8, 1, kernel_size=4, stride=2, padding0),
            ### SIZE: 1*1
            nn.Sigmoid(),
        )

    def _block(self, inChannels, outChannels, kernalSize, stride, padding):
        ### This will be internal blocks of layers of the model
        ### In and out channels are the channel input and output size
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernalSize, 
                      stride, padding, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.discriminator(x)

"""
Generator module
"""
