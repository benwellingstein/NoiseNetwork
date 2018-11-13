import torch.nn as nn
import torch


class DnCNN(nn.Module):
    def __init__(self, channels, layers_num):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features_depth = 64 #Number of feature maps in each layer
        layers = []

        # Add first layer
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features_depth, kernel_size=kernel_size,
                                padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for i in range(layers_num):
            layers.append(nn.Conv2d(in_channels=features_depth, out_channels=features_depth,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features_depth))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(nn.Conv2d(in_channels=features_depth, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, input):
        out = self.dncnn(input) # learning happens here
        #out += input # this is just the return value
        return out
