from __future__ import print_function

import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import utils

# for data loader
import os
from torch.utils.data import Dataset, DataLoader


unloader= transforms.ToPILImage()
def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated


class DnCnnDataset(Dataset):
    def __init__(self, root_dir, noise_stddev, training=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if training:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                #transforms.RandomRotation(180),
                transforms.RandomCrop([80, 80]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()])  # transform it into a torch tensor

        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop([80, 80]),
                transforms.ToTensor()])  # transform it into a torch tensor

        self.stddev = noise_stddev

        img_list = []
        for (root_d, dirs, files) in os.walk(self.root_dir):
            for filename in files:
                if filename.endswith(".jpg"):
                    file_path = os.path.join(root_d, filename)
                    img_list.append(file_path)
        self.img_list = img_list

    @staticmethod
    def get_gaussian_noise(input, stddev, mean=0):
        # Normalize stddev to range [0-1]
        stddev = stddev / 255
        gauss_noise = stddev * torch.randn(input.size()).type(torch.FloatTensor)
        return gauss_noise

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        image = Image.open(image_path)

        # if self.transform:
        # tranform original image
        image_tensor = self.transform(image).squeeze(0)#.unsqueeze(0)
        #imshow(image_tensor)
        noise = DnCnnDataset.get_gaussian_noise(image_tensor, stddev=self.stddev)
        noised_image = image_tensor + noise
        return image_tensor, noise, noised_image
