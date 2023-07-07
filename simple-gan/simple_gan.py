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
            ### This network is binary so 1 output --> Real or fake

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


"""
We now define our hyperparameters
We note that simple GAN's are very sensitive to hyperparameter values 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learningRate = 3e-4     # Best learning rate for adam
zDimention = 64        # Can also try 128 or 256 (multiples of img dims)
imageDimentions = 28*28*1   # 28*28*1 channel
batchSize = 32
nEpochs = 50

"""
We can now define the model functions
"""

discriminator = Discriminaor(imageDimentions).to(device)
generator = Generator(zDimention, imageDimentions).to(device)
fixedNoise = torch.randn((batchSize, zDimention)).to(device)
transformation = transforms.Compose(
    [transforms.ToTensor, transforms.Normalize((0.1307,), (0.3081,))]
)   

dataset = datasets.MNIST(root="dataset/", transform = transformation,
                             download=True)
loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
optimiserGenerator = optim.Adam(generator.parameters(), lr=learningRate)
optimiserDiscriminator = optim.Adam(discriminator.parameters(), lr=learningRate)
criterion = nn.BCELoss()


### For Tensor board
writerFake = SummaryWriter(f"runs/GAN_MNIST/fake") # output fake images
writerReal = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

"""
We now write the model training loop
"""

for epoch in range(nEpochs):
    for batchIdx, (realImages, _) in enumerate(loader):
        realImages = realImages.view(-1, 28*28).to       # resize images
        batchSize = realImages.shape[0]

        ### Train discriminator : max log(D(realImages)) + log(1- D( G(z) ))
        noise = torch.rand(batchSize, zDimention).to(device)    # rand noise tens
        fake_image = generator(noise)                       # G(z)

        discriminatorReal = discriminator(realImages).view(-1)    # D(realImage)
        lossDiscriminatorReal = criterion(discriminatorReal,
                                          torch.ones_like(discriminatorReal))

        discriminatorFake = discriminator(fake).view(-1)          # D( G(z) )
        lossDiscriminatorFake = criterion(discriminatorFake,
                                          torch.zeros_like(discriminatorFake))

        lossDiscriminator = (lossDiscriminatorReal + lossDiscriminatorFake)/2

        discriminator.zero_grad()
        lossDiscriminator.backward()
        optimiserDiscriminator.step()


        ### Train Generator
