import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Generator, Critic, initialiseWeights


"""
Defining hyperparameters
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-4 
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMAGE = 1      # 1 for mnist and 3 for RGB
Z_DIMENTION = 100
N_EPOCH = 8 
FEATURES_CRITIC = 64
FEATURES_GENERATOR = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

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
critic = Critic(CHANNELS_IMAGE, FEATURES_CRITIC).to(device)

initialiseWeights(critic)
initialiseWeights(generator)

optimiserGenerator = optim.RMSprop(generator.parameters(), lr=LEARNING_RATE)
optimiserCritic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

fixedNoise = torch.randn(32, Z_DIMENTION, 1, 1).to(device)

writerReal = SummaryWriter(f"runs_mnist/real")
writerFake = SummaryWriter(f"runs_mnist/fake")
step = 0

### Setting the models to training mode
generator.train()
critic.train()


"""
Training loop
"""

for epoch in range(N_EPOCH):
    for batchIdx, (realImage, _) in enumerate(loader):
        realImage = realImage.to(device)

        for _ in range(CRITIC_ITERATIONS):
        
            ### Training five times: Critic  {max mean(D(x) - mean D(G(z)))}
            noise = torch.randn((BATCH_SIZE, Z_DIMENTION, 1, 1)).to(device)
            fakeImage = generator(noise)

            criticReal = critic(realImage).reshape(-1)
            criticFake = critic(fakeImage).reshape(-1)

            # We write the loss function ourselves
            lossCritic = -(torch.mean(criticReal) - torch.mean(criticFake))           
            
            critic.zero_grad()
            lossCritic.backward(retain_graph=True)
            optimiserCritic.step()

            # Clip our weights after every training loop
            for p in critic.parameters():
                p.data.clamp(-WEIGHT_CLIP, WEIGHT_CLIP)

        ### TRAINing GENERATOR      {min D(G(Z))}
        output = critic(fakeImage).reshape(-1)
        lossGenerator = -torch.mean(output)

        generator.zero_grad()
        lossGenerator.backward()
        optimiserGenerator.step()

        ### Visualise on tensor board
        if batchIdx % 100 == 0:
            print(
                f"Epoch [{epoch}/{N_EPOCH}] Batch {batchIdx}/{len(loader)} \
                  Loss D: {lossCritic:.4f}, loss G: {lossGenerator:.4f}"
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
