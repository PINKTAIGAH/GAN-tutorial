import torch
import torch.nn as nn

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

class Generator(nn.Module):

