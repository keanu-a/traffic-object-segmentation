import numpy as np
from PIL import Image
import torch
from torchvision.datasets import Cityscapes

from utils import get_ignored_classes, get_used_classes, get_used_colors


class CityscapesDataset(Cityscapes):
    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: str = "instance",
        transform=None,
        target_transform=None,
        transforms=None,
    ) -> None:
        super().__init__(
            root, split, mode, target_type, transform, target_transform, transforms
        )

        self.ignored_classes = get_ignored_classes()
        self.used_classes = get_used_classes()
        self.class_colors = get_used_colors()

        self.n_classes = len(self.class_colors)
        self.class_mapping = {
            34: 0,
            7: 1,
            8: 2,
            11: 3,
            12: 4,
            13: 5,
            17: 6,
            19: 7,
            20: 8,
            21: 9,
            22: 10,
            23: 11,
            24: 12,
            25: 13,
            26: 14,
            27: 15,
            28: 16,
            31: 17,
            32: 18,
            33: 19,
        }

    def class_to_color(self, mask):
        rgb_image = torch.zeros(
            (3, mask.size()[0], mask.size()[1]), dtype=torch.uint8
        )
        
        # Loop through all the usuable class colors
        for i in range(self.n_classes):
            m = mask == i
            color = self.class_colors[i]
            
            # Set RGB for image
            for c in range(3):
                rgb_image[c][m] = color[c]

        return rgb_image

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")

        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image), mask=np.array(target))

        image = transformed["image"]
        label = transformed["mask"]

        # Setting label to mask that follows label ids given by cityscapes
        mask = label.clone()
        for v in self.ignored_classes:
            mask[label == v] = 34

        for v in self.used_classes:
            mask[label == v] = self.class_mapping[v]

        # Setting labels to correct color index
        class_mask = mask.clone()
        for v in self.class_mapping:
            class_mask[mask == v] = self.class_mapping[v]

        return image, class_mask
