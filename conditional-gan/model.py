import torch
import torch.nn as nn

"""
Critic model
"""

class Critic(nn.Module):

    def __init__(self, channelImages, featuresD):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
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
            nn.Conv2d(featuresD*8, 1, kernel_size=4, stride=2, padding=0),
            ### SIZE: 1*1
        )

    def _block(self, inChannels, outChannels, kernalSize, stride, padding):
        ### This will be internal blocks of layers of the model
        ### In and out channels are the channel input and output size
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernalSize, 
                      stride, padding, bias=False),
            nn.InstanceNorm2d(outChannels, affine=True), # Layernorm <--> InstanceNorm
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)

"""
Generator module
"""

class Generator(nn.Module):
    def __init__(self, zDimention, channelImages, featuresG):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            ### INPUT SIZE: N * zDimention * 1 * 1
            self._block(zDimention, featuresG*16, 4, 1, 0),
            ### SIZE: 4*4 
            self._block(featuresG*16, featuresG*8, 4, 2, 1),
            ### SIZE: 8*8
            self._block(featuresG*8, featuresG*4, 4, 2, 1),
            ### SIZE: 16*16
            self._block(featuresG*4, featuresG*2, 4, 2, 1),
            ### SIZE: 32*32
            nn.ConvTranspose2d(featuresG*2, channelImages, kernel_size=4,
                               stride=2, padding=1),
            ### SIZE: 64*64
            nn.Tanh(),   # We want output to be in range [-1, 1]
        )

    def _block(self, inChannels, outChannels, kernalSize, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(inChannels, outChannels, kernalSize,
                               stride, padding, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.generator(x)

"""
Initialize weights
"""

def initialiseWeights(model):
    ### Setting init weights when doing one of the following operations
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def testModel():
    N, inChannels, hight, width = 8, 3, 64, 64  # batch of 8, 3 channels, 64*64
    zDimention = 100
    
    x = torch.randn((N, inChannels, hight, width))
    z = torch.randn((N, zDimention, 1, 1))

    critic = Critic(inChannels, 8)
    generator = Generator(zDimention, inChannels, 8)

    initialiseWeights(critic)
    initialiseWeights(generator)

    assert critic(x).shape == (N, 1, 1, 1)  # Check if dims of conv is good
    assert generator(z).shape == (N, inChannels, hight, width)

if __name__ == "__main__":
    testModel()
