import sys
import torch
from unet import UNet
import numpy as np
from PIL import Image
import albumentations as A
from torchvision import transforms, utils
from albumentations.pytorch import ToTensorV2
from cityscapes_dataset import class_to_color

N_CLASSES = 20
TRAINED_MODEL_FILE = "trained_model.pth"
IMAGE_RESIZE = (128, 256)  # Size for resizing images for training

def predict(input_image_name, input_image_path):

  transform = A.Compose(
      [
          A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]),
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

  model = UNet(in_channels=3, class_amount=N_CLASSES)

  model.load_state_dict(torch.load(TRAINED_MODEL_FILE))
  model.eval()

  image = Image.open(input_image_path).convert("RGB")
  image = np.array(image)  # Convert to numpy array
  transformed = transform(image=image)
  image_tensor = transformed['image'].unsqueeze(0)

  with torch.no_grad():
    output = model(image_tensor)

  threshold = torch.zeros(
      (output.size()[0], 3, output.size()[2], output.size()[3])
  )

  for idx in range(0, output.size()[0]):
    maxindex = torch.argmax(output[idx], dim=0).cpu().int()
    threshold[idx] = class_to_color(maxindex)

  image = inv_norm(image_tensor)

  utils.save_image(image, f"./pred_images/{input_image_name}_og.png")
  utils.save_image(threshold, f"./pred_images/{input_image_name}_pred.png")


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Please provide a file name")
  else:
    input_image_name = sys.argv[1]
    input_image_path = f"./test_images/{input_image_name}"

    predict(input_image_name, input_image_path)