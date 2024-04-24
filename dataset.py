import os
from torch.utils.data import Dataset
from torch import from_numpy
from PIL import Image


class CityscapesDataset(Dataset):
    def __init__(self, image_dir, label_dir, split, image_transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.split = split # "train", "val", or "test"
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.image_paths = []
        self.label_paths = []

        # Load image and label paths based on split
        self._load_paths()


    # Iterates through directories based on the split
    def _load_paths(self):
        # Go to specfied split directory and image file paths to image_paths
        for dirpath, _, filenames in os.walk(os.path.join(self.image_dir, self.split)):
            for filename in filenames:
                if filename.endswith(".png"):
                    image_path = os.path.join(dirpath, filename)

                    # Only using labelIds since doing semantic segmentation
                    label_dirpath = dirpath.replace("leftImg8bit", "gtFine")
                    label_filename = filename.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    label_path = os.path.join(label_dirpath, label_filename)

                    self.image_paths.append(image_path)
                    self.label_paths.append(label_path)

    
    # Gets length of the dataset
    def __len__(self):
        return len(self.image_paths)
    

    # Gets a specific image-label based on index
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        return image, label