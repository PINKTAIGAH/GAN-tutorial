import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, rootDirectory):
        self.rootDirectory = rootDirectory
        self.listFiles = os.listdir(self.rootDirectory)

    def __len__(self):
        return len(self.listFiles)

    def __getitem__(self, index):
        imageFile = self.listFiles[index]
        imagePath = os.path.join(self.rootDirectory, imageFile)
        image = np.array(Image.open(imagePath))
        inputImage = image[:, :600, :]
        targetImage = image[:, 600:, :]

        augmentations = config.bothTransform(image=inputImage, image0=targetImage)
        inputImage = augmentations["image"]
        targetImage = augmentations["image0"]

        inputImage = config.transformOnlyInput(image=inputImage)["image"]
        targetImage = config.transformOnlyMask(image=targetImage)["image"]

        return inputImage, targetImage


if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
