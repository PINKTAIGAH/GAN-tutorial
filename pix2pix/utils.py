import torch
import config
from torchvision.utils import save_image

def saveSomeExamples(generator, validationLoader, epoch, folder):
    x, y = next(iter(validationLoader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    generator.eval()
    with torch.no_grad():
        yFake = generator(x)
        yFake = yFake * 0.5 + 0.5  # remove normalization#
        save_image(yFake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    generator.train()


def saveCheckpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def loadCheckpoint(checkpointFile, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpointFile, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for paramGroup in optimizer.paramGroups:
        paramGroup["lr"] = lr
