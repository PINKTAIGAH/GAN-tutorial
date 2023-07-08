import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Generator, Discriminator, initialiseWeights


"""
Defining hyperparameters
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMAGE = 1      # 1 for mnist and 3 for RGB
Z_DIMENTION = 100
N_EPOCH = 8 
FEATURES_DISCRIMINATIOR = 64
FEATURES_GENERATOR = 64

transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMAGE)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMAGE)],
    ),
])

dataset = datasets.MNIST(root="../dataset/", train=True, transform= transforms,
                         download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
generator = Generator(Z_DIMENTION, CHANNELS_IMAGE, FEATURES_GENERATOR).to(device)
discriminator = Discriminator(CHANNELS_IMAGE, FEATURES_DISCRIMINATIOR).to(device)

initialiseWeights(discriminator)
initialiseWeights(generator)

### Betas are obtained form the original DCGAN paper
optimiserGenerator = optim.Adam(generator.parameters(), lr=LEARNING_RATE,
                                betas=(0.5, 0.999))
optimiserDiscriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE,
                                betas=(0.5, 0.999))
criterion = nn.BCELoss()
fixedNoise = torch.randn(32, Z_DIMENTION, 1, 1).to(device)

writerReal = SummaryWriter(f"runs_mnist/real")
writerFake = SummaryWriter(f"runs_mnist/fake")
step = 0

### Setting the models to training mode
generator.train()
discriminator.train()


"""
Training loop
"""

for epoch in range(N_EPOCH):
    for batchIdx, (realImage, _) in enumerate(loader):
        realImage = realImage.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIMENTION, 1, 1)).to(device)
        fakeImage = generator(noise)
        
        ### Training Discriminator  {max log(D(x)) + log(1-D(G(z)))}
        discriminatorReal = discriminator(realImage).reshape(-1)
        discriminatorFake = discriminator(fakeImage).reshape(-1)

        lossDiscriminatorReal = criterion(discriminatorReal, 
                                          torch.ones_like(discriminatorReal))
        lossDiscriminatorFake = criterion(discriminatorFake,
                                          torch.zeros_like(discriminatorFake))
        lossDiscriminator = (lossDiscriminatorReal + lossDiscriminatorFake)/2 

        discriminator.zero_grad()
        lossDiscriminator.backward(retain_graph=True)   # to reuse for generator
        optimiserDiscriminator.step()

        ### TRAINing GENERATOR      {max log(D(G(z)))}
        output = discriminator(fakeImage).reshape(-1)
        lossGenerator = criterion(output, torch.ones_like(output))
        
        generator.zero_grad()
        lossGenerator.backward()
        optimiserGenerator.step()

        ### Visualise on tensor board
        if batchIdx % 100 == 0:
            print(
                f"Epoch [{epoch}/{N_EPOCH}] Batch {batchIdx}/{len(loader)} \
                  Loss D: {lossDiscriminator:.4f}, loss G: {lossGenerator:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixedNoise)
            
                ### Take out (up to) 32 examples
                imageGridReal = torchvision.utils.make_grid(
                    realImage[:32], normalize=True
                )
                imageGridFake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writerReal.add_image("real", imageGridReal, global_step=step)
                writerFake.add_image("fake", imageGridFake, global_step=step)
            
                step += 1

writerFake.close()
writerReal.close()
