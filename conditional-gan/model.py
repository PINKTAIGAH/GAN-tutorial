import torch
import torch.nn as nn
"""
By adding an extra channel in the latend nouse vector and the input image, we 
are also forcing the model to lkearn what class the input image is and to
copy that image when trying to generate a replica
"""

"""
Critic model
"""

class Critic(nn.Module):

    def __init__(self, channelImages, featuresD, nClasses, imageSize):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            ### INPUT SIZE: N * channelImages + 1 (embedding) * 64 * 64
            nn.Conv2d(channelImages+1, featuresD, kernel_size=4,
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
        self.imageSize = imageSize
        self.embed = nn.Embedding(nClasses, self.imageSize*self.imageSize)

    def _block(self, inChannels, outChannels, kernalSize, stride, padding):
        ### This will be internal blocks of layers of the model
        ### In and out channels are the channel input and output size
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernalSize, 
                      stride, padding, bias=False),
            nn.InstanceNorm2d(outChannels, affine=True), # Layernorm <--> InstanceNorm
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1,
                                            self.imageSize, self.imageSize)
        ### Concatinate embedding to the x tensor to include info of img class
        ### Embedding acts like an additional channel of the image
        x = torch.cat([x, embedding], dim=1)
        return self.critic(x)

"""
Generator module
"""

class Generator(nn.Module):
    def __init__(self, zDimention, channelImages, featuresG,
                 nClasses, imageSize, embedSize):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            ### INPUT SIZE: N * zDimention + embed_size * 1 * 1
            self._block(zDimention + embedSize, featuresG*16, 4, 1, 0),
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
        
        self.imageSize = imageSize
        self.embed = nn.Embedding(nClasses, embedSize)

    def _block(self, inChannels, outChannels, kernalSize, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(inChannels, outChannels, kernalSize,
                               stride, padding, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(),
        )
    
    def forward(self, x, labels):
        ### latent vector Z: N * noise_dimention * 1 * 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3) #unsquezze adds 1*1 
        x = torch.cat([x, embedding], dim=1)
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
