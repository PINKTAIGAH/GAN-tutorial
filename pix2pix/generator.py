import torch
import torch.nn as nn
from torch.nn.modules import activation

"""
Write generic convolution block used for the upscaling and downscaling of the 
encoder/decoder sections
"""

class Block(nn.Module):
    
    def __init__(self, inChannel, outChannels, down=True,
                 activation="relu", useDropout=False):
    ### Down refers to if we are in the encoding or decoding phase os the u-net
        super().__init__()
        self.convolution = nn.Sequential(
            
            nn.Conv2d(inChannel, outChannels, kernel_size=4, stride=2,
                      padding=1, bias=False , padding_mode="reflect")\
            if down ### Use conv2d if in encoder section
            else nn.ConvTranspose2d(inChannel, outChannels, kernel_size=4,
                                    stride=2, padding=1, bias=False),
            ### Use convtransose2d if in decoder section 
            
            nn.BatchNorm2d(outChannels),
            nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2),
        )
        self.useDropout = useDropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.convolution(x)
        return self.dropout(x) if self.useDropout else x

"""
Define the generator class
"""

class Generator(nn.Module):
    def __init__(self, inChannels, features=64):
        super().__init__()
        self.initialDown = nn.Sequential(
            nn.Conv2d(inChannels, features, kernel_size=4, stride=2, 
                      padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
        )   # 128

        self.down1 = Block(features, features*2, down=True,
                           activation="leaky", useDropout=False)    # 64
        self.down2 = Block(features*4, features*8, down=True,
                           activation="leaky", useDropout=False)    # 32
        self.down3 = Block(features*8, features*8, down=True,
                           activation="leaky", useDropout=False)    # 16
        self.down4 = Block(features*8, features*8, down=True,
                           activation="leaky", useDropout=False)    # 8
        self.down5 = Block(features*8, features*8, down=True,
                           activation="leaky", useDropout=False)    # 5
        self.down6 = Block(features*8, features*8, down=True,
                           activation="leaky", useDropout=False)    # 2

        # Final layer of the encoder
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, 
                      padding=1, padding_mode="reflect"),
            nn.ReLU(),
        )
