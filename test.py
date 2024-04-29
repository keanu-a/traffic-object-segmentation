import os
import torch
from torch import cuda
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from unet import UNet

CITYSCAPES_PATH = os.path.join("cityscapes")
IMAGE_DIR = os.path.join(CITYSCAPES_PATH, "leftImg8bit")
LABEL_DIR = os.path.join(CITYSCAPES_PATH, "gtFine")

BATCH_SIZE = 16  # Try 32 next
IMAGE_SIZE = (256, 256)  # Size for resizing images for training
N_CLASSES = 19  # Cityscapes has 19 classes, dataset has 30
DEVICE = "cuda" if cuda.is_available() else "cpu"
TRAINED_MODEL_FILE = "trained_model.pth"


def calc_iou(pred, label):
    intersection = (pred == label).sum().float()
    union = (pred + label).sum().float() - (pred == label).sum().float()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def test():
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),  # Resizing for training
            transforms.ToTensor(),  # Converting to PyTorch tensors
        ]
    )

    validation_set = Cityscapes(
        "./cityscapes",
        split="val",
        mode="fine",
        target_type="semantic",
        transform=transform,
        target_transform=transform,
    )

    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
    print("Validation data loaded")

    model = UNet(in_channels=3, class_amount=N_CLASSES)

    model.load_state_dict(torch.load(TRAINED_MODEL_FILE))

    model.eval()

    iou_score = 0

    with torch.no_grad():
        for image, label in tqdm(validation_loader):
            image, label = image.to(DEVICE), label.to(DEVICE)

            output = model(image)

            # Convert prediction probabilities to class labels
            pred_labels = torch.argmax(output, dim=1)

            # Calculate IoU for each image in the batch
            iou_score += calc_iou(pred_labels, label)

        iou_score = iou_score / len(validation_loader)
        print(f"Average IoU: {iou_score}")


if __name__ == "__main__":
    test()
