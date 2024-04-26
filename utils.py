import os
from dataset import CityscapesDataset
from torch.utils.data import DataLoader
import torch

CITYSCAPES_PATH = os.path.join("cityscapes")
IMAGE_DIR = os.path.join(CITYSCAPES_PATH, "leftImg8bit")
LABEL_DIR = os.path.join(CITYSCAPES_PATH, "gtFine")


def get_dataloaders(
    image_dir, label_dir, batch_size, image_transform, label_transform, pin_memory
):
    pass


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = torch.sigmoid(model(x))
