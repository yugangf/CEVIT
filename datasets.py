import math
import os
import torch
import json
import torchvision
import numpy as np
import skimage.io

from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
from torchvision import transforms as pth_transforms

# Image transformation applied to all images
transform = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class ImageDataset:
    def __init__(self, image_path, resize=None):

        self.image_path = image_path
        self.name = image_path.split("/")[-1]

        # Read the image
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # Build a dataloader
        if resize is not None:
            transform_resize = pth_transforms.Compose(
                [
                    pth_transforms.ToTensor(),
                    pth_transforms.Resize(resize),
                    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            img = transform_resize(img)
            self.img_size = list(img.shape[-1:-3:-1])
        else:
            img = transform(img)
            self.img_size = list(img.shape[-1:-3:-1])
        self.dataloader = [[img, image_path]]

    def get_image_name(self, *args, **kwargs):
        return self.image_path.split("/")[-1].split(".")[0]

    def load_image(self, *args, **kwargs):
        return Image.open(self.image_path).convert("RGB").resize(self.img_size)


class Dataset:
    def __init__(self, dataset_name, dataset_set, remove_hards):
        """
        Build the dataloader
        """
        self.dataset_name = dataset_name
        self.set = dataset_set
        if dataset_name == "mydata":
            # self.root_path = "D:/Py project/Database/test/ISR_15/"
            self.root_path = "D:/Py project/Database/val_separate/val_1500/"
            # self.root_path = "D:/Py project/Database/sentinel/"
            self.year = "2023"
        else:
            raise ValueError("Unknown dataset.")
        if not os.path.exists(self.root_path):
            raise ValueError("Please setup the path to datasets.")
        self.name = f"{self.dataset_name}_{self.set}"
        self.image_names = [f for f in os.listdir(os.path.join(self.root_path, "Images")) if f.endswith(".jpg")]
        self.image_names.sort()
        self.image_index = 0
        # print(self.image_names)

        if "mydata" in dataset_name:
            self.dataloader = torchvision.datasets.ImageFolder(
                self.root_path,
                transform=transform,

            )
        else:
            raise ValueError("Unknown dataset.")

        self.remove_hards = remove_hards
        self.hards = []
        if remove_hards:
            self.name += f"-nohards"
            self.hards = self.get_hards()
            print(f"Nb images discarded {len(self.hards)}")

    def load_image(self, im_name):
        """
        Load the image corresponding to the im_name
        """
        if "mydata" in self.dataset_name:
            image_path = os.path.join(self.root_path, "Images", im_name)
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 检查文件是否存在且是图像文件
                image = skimage.io.imread(image_path)
            else:
                raise ValueError(f"Invalid image file: {image_path}")
            return image

    def get_image_name(self, inp):
        if self.image_index < len(self.image_names):
            image_name = self.image_names[self.image_index]
            self.image_index += 1
            return image_name
        else:
            return None