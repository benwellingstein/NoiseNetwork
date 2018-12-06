from dataset import DnCnnDataset
from models import DnCNN
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.utils

import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import pickle
from skimage import measure
import numpy as np

unloader= transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()  # we clone the tensor to not do changes on it - add .detach()
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

    fig = plt.figure()
    for i in range (1, cols*rows+1):
        image = images[i-1].cpu().clone()#.squeeze(0)
        #image = torch.FloatTensor(1, 80, 80)
        image = unloader(image)
        title = titles[i-1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(image)
        plt.title(title)
    plt.show()


def save_model(model, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}
        , file_path)



# Main
saveModelPath = "/home/osherm/PycharmProjects/NoiseNetwork/model.pth"
trainRootPath = "/home/osherm/PycharmProjects/NoiseNetwork/train"
evalSetPath = "/home/osherm/PycharmProjects/NoiseNetwork/test/BSD68/"
trainingEvalPath = "/home/osherm/PycharmProjects/NoiseNetwork/trainingEvals/"

noise_stddev = 25
epochs_num = 10
batch_size = 32
lr = 10e-2

log = open('log.txt', 'a') 

log.write("--------------------------------------------------------------")
log.write("Starting new network")
log.write("Using images from {}".format(evalSetPath))
log.write("Number of Epochs: {}, Batch size: {}".format(epochs_num, batch_size))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


training_set = DnCnnDataset(trainRootPath, noise_stddev, training=True)
training_generator = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)

eval_set = DnCnnDataset(evalSetPath, noise_stddev, training=False, evaluate=False)
eval_set_generator = DataLoader(eval_set,batch_size=batch_size, shuffle=False, drop_last=True)


DnCnn_net = DnCNN(channels=batch_size, layers_num=10)
#transfer to GPU - cuda/to.device

criterion = nn.MSELoss()

# Move to GPU
model = DnCnn_net.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
optim_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(0.1*epochs_num), int(0.9*epochs_num)], gamma=0.1)


psnr_values = []
for epoch in range(epochs_num):
    running_loss = 0.0
    optim_scheduler.step()

    for i, data in enumerate(training_generator):
        image, noise, noised_image = data

        image, noise, noised_image = image.to(device).unsqueeze(0),\
                                     noise.to(device).unsqueeze(0),noised_image.unsqueeze(0).to(device)

        optimizer.zero_grad()

        outputs = model(noised_image)
        loss = criterion(outputs,image)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i%10 == 9:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i+1, running_loss/10))
            running_loss = 0.0

    if epoch%10 == 9:
        save_model(model, saveModelPath)
        # Sanity check - calculate PSNR
        # TODO: display/save images
        # TODO: log psnr values along iterations


        # Calculate average PSNR for eval dataset
        eval_psnr_sum = 0.0
        for j, eval_data in enumerate(eval_set_generator):
            eval_img_batch, _, eval_noised_img_batch = eval_data
            eval_img_batch, eval_noised_img_batch = eval_img_batch.to(device).unsqueeze(0), eval_noised_img_batch.to(device).unsqueeze(0)
            cleaned_eval_img_batch = model(eval_noised_img_batch)

            #Calculate average PSNR for the batch
            batch_psnr_sum = 0.0
            for sample_ind in range(batch_size):
                eval_img = eval_img_batch.detach().cpu().numpy()[0,sample_ind,:,:]
                eval_cleaned_img = cleaned_eval_img_batch.detach().cpu().numpy()[0,sample_ind,:,:]
                psnr_val = measure.compare_psnr(eval_img, eval_cleaned_img)
                eval_psnr_sum += psnr_val
                if (j==0 and sample_ind==0):
                    # Save first image to display progress during training
                    I8 = (((eval_cleaned_img - eval_cleaned_img.min()) / (eval_cleaned_img.max() - eval_cleaned_img.min())) * 255.9).astype(np.uint8)
                    img = Image.fromarray(I8)
                    img.save(trainingEvalPath+"img0.png")


        avg_psnr = eval_psnr_sum / eval_set.__len__()
        psnr_values.append(avg_psnr)
        # average psnr sum over batch size
        log.write("psnr value for epoch {} is {}".format(epoch, avg_psnr))
        print("average psnr is {}".format(psnr_values))

print("Finished training")
save_model(model, saveModelPath)
log.close()


# TODO : test loading model state
checkpoint = torch.load(saveModelPath)
#start_epoch = checkpoint['epoch']
#best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])


print("Start testing")
testRootPath = "/home/osherm/PycharmProjects/NoiseNetwork/val"
test_set = DnCnnDataset(testRootPath, noise_stddev, training=False)
test_generator = DataLoader(test_set, batch_size=batch_size, num_workers=4,drop_last=True)

for i, data in enumerate(test_generator, 0):
    image, noise, noised_image = data

    image, noise, noised_image = image.to(device).unsqueeze(0), \
                                 noise.to(device).unsqueeze(0), noised_image.unsqueeze(0).to(device)
    #learned_noise = model(noised_image)
    #clean_image = noised_image[:,0,:,:] - learned_noise[:,0,:,:]

    clean_image = model(noised_image)[:,0,:,:]

    images_for_display = [image[:,0,:,:], noised_image[:,0,:,:], clean_image]
    titles = ["Original image", "Noised image", "Clean image"]
    #plt.figure(i)
    subplot(images_for_display, titles, 1, 3)
