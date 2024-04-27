import os
from torch.utils.data import Dataset
from PIL import Image


class CityscapesDataset(Dataset):
    def __init__(self, image_dir, label_dir, split, i_transform=None, l_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.split = split  # "train", "val", or "test"
        self.i_transform = i_transform
        self.l_transform = l_transform
        self.images = []
        self.labels = []

        # Load image and label paths based on split
        self.load_cityscapes()

    # Iterates through directories based on the split
    def load_cityscapes(self):
        # Go to specfied split directory and image file paths to images
        for dirpath, _, filenames in os.walk(os.path.join(self.image_dir, self.split)):
            for filename in filenames:
                if filename.endswith(".png"):
                    image = os.path.join(dirpath, filename)

                    # Only using labelIds since doing semantic segmentation
                    label_dirpath = dirpath.replace("leftImg8bit", "gtFine")
                    label_filename = filename.replace(
                        "_leftImg8bit.png", "_gtFine_labelIds.png"
                    )
                    label = os.path.join(label_dirpath, label_filename)

                    self.images.append(image)
                    self.labels.append(label)

    # Gets length of the dataset
    def __len__(self):
        return len(self.images)

    # Gets a specific image-label based on index
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image).convert("RGB")
        label = Image.open(label).convert("L")

        # Normalizes images as well
        if self.i_transform:
            image = self.i_transform(image)

        if self.l_transform:
            label = self.l_transform(label)
            label = label.long()  # Need to convert to type Long for CrossEntropyLoss

        return image, label
