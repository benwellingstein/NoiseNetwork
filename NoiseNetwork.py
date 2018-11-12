from __future__ import print_function



"""


The great list of TODOS


    open git - BEN
    
    data loader
    
    build network 
        
        loss layer
        
    training 

    matlab running and refrencing 
    
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import utils


# for data loader
import os
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def gaussian(input, stddev, mean = 0):
    if stddev > 1:
        stddev = stddev / 255
    if (input.is_cuda):
        gauss_noise = input.data.new_empty(input.size()).normal_(mean, stddev)
            #stddev * torch.randn(input.size()).type(torch.cuda.FloatTensor)
    else:
        gauss_noise = input.data.new_empty(input.size()).normal_(mean, stddev)#.type(torch.FloatTensor)
    #
    return gauss_noise#.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(3) # pause a bit so that plots are updated


class DataSetLoader(Dataset):
    def __init__(self, root_dir, noise_stddev, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.stddev = noise_stddev

        img_list = []
        for (root_d, dirs, files) in os.walk(self.root_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                img_list.append(file_path)

        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(image_name)

        #if self.transform:
        # tranform original image
        image_tensor = self.transform(image).unsqueeze(0).to(device, torch.float)
        noise = gaussian(image_tensor, stddev = self.stddev)
        noised_image = image_tensor + noise
        sample = {'image': image_tensor, 'noise': noise, 'noised_image': noised_image}

        return sample

    @staticmethod
    def get_gaussian_noise(input, stddev, mean = 0):
        # Normalize stddev to range [0-1]
        if stddev > 1:
            stddev = stddev/255
        if input.is_cuda:
            gauss_noise = stddev * torch.randn(input.size()).type(torch.cuda.FloatTensor)
        else:
            gauss_noise = stddev * torch.randn(input.size()).type(torch.FloatTensor)
        # input.data.new_empty(input.size()).normal_(mean,sttdev)
        return gauss_noise


loader_transforms = transforms.Compose([
      transforms.Grayscale(),
      transforms.RandomCrop([80,80], padding=10),
      transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

root = "/home/osherm/BSD/BSR/BSDS500/data/images/test/"


bsd_dataloader= DataSetLoader(root, noise_stddev=10, transform=loader_transforms)

'''
# Meant for testing dataloader
sample_img = bsd_dataloader.__getitem__(2)

plt.figure()
imshow(sample_img['noised_image'])
'''
