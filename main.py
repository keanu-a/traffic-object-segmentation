import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CityscapesDataset
import matplotlib.pyplot as plt

CITYSCAPES_PATH = os.path.join("cityscapes")
IMAGE_DIR = os.path.join(CITYSCAPES_PATH, "leftImg8bit")
LABEL_DIR = os.path.join(CITYSCAPES_PATH, "gtFine")

BATCH_SIZE = 8 # Number of images to be processed together during training
IMAGE_SIZE = (256, 256) # Size for resizing images for training
CLASS_AMOUNT = 30 # Cityscapes has 30 classes

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE), # Resizing for training
    transforms.ToTensor() # Converting to PyTorch tensors
])

# Creating training and validation datasets
train_dataset = CityscapesDataset(image_dir=IMAGE_DIR, label_dir=LABEL_DIR, split="train", transform=transform)

image, label = train_dataset[0]

to_pil = transforms.ToPILImage()
image_pil = to_pil(image)
label_pil = to_pil(label)

# Display the image and label
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(image_pil)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Label")
plt.imshow(label_pil, cmap='gray')
plt.axis('off')

plt.show()