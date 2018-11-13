from dataset import DnCnnDataset
from models import DnCNN
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import pickle

unloader= transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it - add .detach()
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
    if len(images) != rows*cols or len(images)!=len(titles):
        raise("Error - number of images isn't equal to the number of sublots")

    #fig = plt.figure()
    for i in range (1, cols*rows+1):
        image = images[i-1].cpu().clone()#.squeeze(0)
        #image = torch.FloatTensor(1, 80, 80)
        image = unloader(image)
        title = titles[i-1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(image)
        plt.title(title)
    plt.show()


def save_models_params(params, file_path):
    with open (file_path, "wb") as f:
        pickle.dump(params,f)
        print("Parameters were saved")



# Main
trainRootPath = "/home/osherm/PycharmProjects/NoiseNetwork/train"
noise_stddev = 25
epochs_num = 25
batch_size = 4


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


training_set = DnCnnDataset(trainRootPath, noise_stddev, training=True)
training_generator = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=4)

DnCnn_net = DnCNN(channels=batch_size, layers_num=10)
#transfer to GPU - cuda/to.device

criterion = nn.MSELoss()

# Move to GPU
model = DnCnn_net.to(device)

#optimizer = optim.SGD(DnCnn_net.parameters(), lr=0.00001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

for epoch in range(epochs_num):
    running_loss = 0.0

    for i, data in enumerate(training_generator):
        image, noise, noised_image = data

        image, noise, noised_image = image.to(device).unsqueeze(0),\
                                     noise.to(device).unsqueeze(0),noised_image.unsqueeze(0).to(device)

        optimizer.zero_grad()

        outputs = model(noised_image)
        loss = criterion(outputs,noise) #criterion(outputs, image)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%10 == 9:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i+1, running_loss/10))
            running_loss = 0.0

print("Finished training")

print("Start testing")
testRootPath = "/home/osherm/PycharmProjects/NoiseNetwork/val"
test_set = DnCnnDataset(testRootPath, noise_stddev, training=False)
test_generator = DataLoader(test_set, batch_size=4, num_workers=4)

for i, data in enumerate(test_generator, 0):
    image, noise, noised_image = data

    image, noise, noised_image = image.to(device).unsqueeze(0), \
                                 noise.to(device).unsqueeze(0), noised_image.unsqueeze(0).to(device)
    learned_noise = model(noised_image)
    clean_image = noised_image[:,0,:,:] - learned_noise[:,0,:,:]

    #clean_image = model(noised_image)[:,0,:,:]

    images_for_display = [image[:,0,:,:], noised_image[:,0,:,:], clean_image]
    titles = ["Original image", "Noised image", "Clean image"]
    plt.figure(i)
    subplot(images_for_display, titles, 1, 3)
