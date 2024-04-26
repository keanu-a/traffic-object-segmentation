import os
import torch.nn as nn
import torch
from torch import optim, cuda
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CityscapesDataset
from unet import UNet

CITYSCAPES_PATH = os.path.join("cityscapes")
IMAGE_DIR = os.path.join(CITYSCAPES_PATH, "leftImg8bit")
LABEL_DIR = os.path.join(CITYSCAPES_PATH, "gtFine")

BATCH_SIZE = 16  # Number of images to be processed together during training (32?)
IMAGE_HEIGHT = 256
IMAGE_LENGTH = 256
IMAGE_SIZE = (256, 256)  # Size for resizing images for training
CLASS_AMOUNT = 30  # Cityscapes has 30 classes
EPOCHS = 10  # 2
LEARNING_RATE = 3e-4
DEVICE = "cuda" if cuda.is_available() else "cpu"


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for idx, (image, label) in enumerate(loop):
        data = image.to(DEVICE)
        label = label.float().unsqueeze(1).to(DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            pred = model(image)
            loss = loss_fn(pred, label)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    image_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_LENGTH),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.HorizontalFlip(),
            ToTensorV2,
        ]
    )

    label_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_LENGTH),
            ToTensorV2,
        ]
    )

    model = UNet(in_channels=3, out_channels=CLASS_AMOUNT).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()  # Since multiple classes
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Creating training and validation datasets
    training_set = CityscapesDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        split="train",
        image_transform=image_transform,
        label_transform=label_transform,
    )
    validation_set = CityscapesDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        split="val",
        image_transform=image_transform,
        label_transform=label_transform,
    )

    # Setting up dataloaders
    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        train(training_loader, model, optimizer, loss_fn, scaler)

        # save model

        # check accuracy

        # print examples to folder


image_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),  # Resizing for training
        transforms.ToTensor(),  # Converting to PyTorch tensors
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # mean and std from ImageNet - good practice
    ]
)

label_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),  # Resizing for training
        transforms.ToTensor(),  # Converting to PyTorch tensors
    ]
)

# for epoch in tqdm(range(EPOCHS)):
# model.train()

#     train_running_loss = 0
#     val_running_loss = 0

#     for idx, (image, label) in enumerate(tqdm(training_loader)):
#         img = image.float().to(DEVICE)
#         mask = label.float().to(DEVICE)

#         pred = model(img)
#         optimizer.zero_grad()

#         loss = criterion(pred, mask)
#         train_running_loss += loss.item()

#         loss.backward()
#         optimizer.step()

#     train_loss = train_running_loss / (idx + 1)

#     model.eval()

#     with torch.no_grad():
#         for idx, (image, label) in enumerate(tqdm(validation_loader)):
#             img = image.float().to(DEVICE)
#             mask = label.float().to(DEVICE)

#             pred = model(img)
#             loss = criterion(pred, mask)

#             val_running_loss += loss.item()

#         val_loss = val_running_loss / (idx + 1)

#     print("-"*30)
#     print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
#     print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
#     print("-"*30)
