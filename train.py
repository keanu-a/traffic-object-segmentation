import os
import torch
import torch.nn as nn
from torch import optim, cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

from torchvision import transforms, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

from unet import UNet
from cityscapes_dataset import CityscapesDataset

CITYSCAPES_PATH = os.path.join("cityscapes")
IMAGE_DIR = os.path.join(CITYSCAPES_PATH, "leftImg8bit")
LABEL_DIR = os.path.join(CITYSCAPES_PATH, "gtFine")
CHECKPOINT_DIR = os.path.join("checkpoints")

BATCH_SIZE = 16  # Number of images to be processed together during training (32?)
IMAGE_SIZE = (1024, 2048)  # Size of original image
IMAGE_RESIZE = (128, 256)  # Size for resizing images for training
EPOCHS = 3
LEARNING_RATE = 0.001
DEVICE = "cuda" if cuda.is_available() else "cpu"


def train():
    transform = A.Compose(
        [
            A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Weights from ImageNet
            ToTensorV2(),
        ]
    )

    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )

    # Initializing datasets from Cityscapes class form torchvision
    # training_set = Cityscapes(
    training_set = CityscapesDataset(
        "./cityscapes",
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=transform,
    )

    n_classes = training_set.n_classes

    # Initializing the model
    model = UNet(in_channels=3, class_amount=n_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()  # Since dealing with classes
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Setting up dataloaders
    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

    # Setting up the training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        print(f"EPOCH: {epoch} - Training...")

        # Run model on training set
        for image, label in tqdm(training_loader):
            optimizer.zero_grad()

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)

            label_t = label.squeeze(1).long()

            loss = loss_fn(output, label_t)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Creating an image from the output
        threshold = torch.zeros(
            (output.size()[0], 3, output.size()[2], output.size()[3])
        )

        for idx in range(0, output.size()[0]):
            maxindex = torch.argmax(output[idx], dim=0).cpu().int()
            threshold[idx] = training_set.class_to_color(maxindex)
            
        image = inv_norm(image)

        utils.save_image(image, f"./result/image_{epoch}_{epoch + 1}_image.png")
        utils.save_image(threshold, f"./result/pred_{epoch}_{epoch + 1}_image.png")
        

        # Calculate losses from training on training and validation sets
        train_loss_average = train_loss / len(training_loader)

        print(f"EPOCH: {epoch} - Train loss: {train_loss_average}")

        # Saving checkpoint after every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "average_train_loss": train_loss_average,
                },
                checkpoint_path,
            )

    # Saving final trained model
    torch.save(model.state_dict(), "trained_model.pth")


if __name__ == "__main__":
    train()
