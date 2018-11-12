from dataset import DnCnnDataset
from models import DnCNN
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms



unloader= transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)

def subplot(images, titles, rows, cols):
    '''

    :param images: an array of tensors representing the images
    :param rows: number of rows in the grid
    :param cols: number of cols in the grid
    :return:
    '''
    if len(images) != rows*cols:
        raise("Error - number of images isn't equal to the number of sublots")

    fig = plt.figure()
    for i in range (1, cols*rows+1):
        image = images[i-1].squeeze(0)
        #image = torch.FloatTensor(1, 80, 80)
        image = unloader(image)
        title = titles[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(image)
        plt.title(title)
    plt.show()



trainRootPath = "/home/osherm/PycharmProjects/NoiseNetwork/train"
noise_stddev = 50
epochs_num = 2


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


training_set = DnCnnDataset(trainRootPath, noise_stddev)
training_generator = DataLoader(DnCnnDataset, batch_size=4, shuffle=True)

DnCnn_net = DnCNN(channels=1, layers_num=10)

criterion = nn.MSELoss()

# Move to GPU
device_ids = [0]
model = nn.DataParallel(DnCnn_net, device_ids=device_ids).cuda()

optimizer = optim.SGD(DnCnn_net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(1):
    running_loss = 0.0

    for i, data in enumerate(training_set,0):
        image, noise, noised_image = data

        optimizer.zero_grad()

        outputs = model(noised_image)
        loss = criterion(outputs, noise)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%10 == 9:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i+1, running_loss/10))
            running_loss = 0.0

print("Finished training")

print("Start testing")
testRootPath = "/home/osherm/PycharmProjects/NoiseNetwork/test"
testLoader = DnCnnDataset(testRootPath, noise_stddev)

for i, data in enumerate(testLoader, 0):
    if i==0:
        image, noise, noised_image = data

        learned_noise = model(noised_image)
        clean_image = noised_image - learned_noise

        images_for_display = [image, noised_image, clean_image]
        titles = ["Original image", "Noised image", "Clean image"]
        plt.figure(i)
        subplot(images_for_display, titles,1,3)
    else:
        break
