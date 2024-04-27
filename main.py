import os
import torch.nn as nn
import torch
from torch import optim, cuda
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import CityscapesDataset
from unet import UNet

CITYSCAPES_PATH = os.path.join("cityscapes")
IMAGE_DIR = os.path.join(CITYSCAPES_PATH, "leftImg8bit")
LABEL_DIR = os.path.join(CITYSCAPES_PATH, "gtFine")

BATCH_SIZE = 16  # Number of images to be processed together during training (32?)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (256, 256)  # Size for resizing images for training
N_CLASSES = 19  # Cityscapes has 19 classes, dataset has 30
EPOCHS = 10  # 2
LEARNING_RATE = 0.001  # 3e-4
DEVICE = "cuda" if cuda.is_available() else "cpu"


def main():
    i_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),  # Resizing for training
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converting to PyTorch tensors
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # mean and std from ImageNet - good practice
        ]
    )

    l_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),  # Resizing for training
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converting to PyTorch tensors
        ]
    )

    # Creating training and validation datasets
    training_set = CityscapesDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        split="train",
        i_transform=i_transform,
        l_transform=l_transform,
    )

    validation_set = CityscapesDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        split="val",
        i_transform=i_transform,
        l_transform=l_transform,
    )

    # Creating the model
    model = UNet(in_channels=3, class_amount=N_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()  # Since multiple classes
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Setting up dataloaders
    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

    inverse_transform = transforms.Compose(
        [
            transforms.Normalize(
                (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
                (1 / 0.229, 1 / 0.224, 1 / 0.225),
            )
        ]
    )

    # Setting up the training loop
    for i in range(EPOCHS):
        model.train()
        train_loss = 0
        val_loss = 0

        for image, label in tqdm(training_loader):
            optimizer.zero_grad()
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)

            label = label.squeeze(1)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        for image, label in tqdm(validation_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)

            label = label.squeeze(1)  # Labels need to be 3D
            loss = loss_fn(output, label)

            val_loss += loss.item()

        train_loss_average = train_loss / len(training_loader)
        val_loss_average = val_loss / len(validation_loader)

        print(f"Train loss: {train_loss_average}, Validation loss: {val_loss_average}")

        break


if __name__ == "__main__":
    main()
