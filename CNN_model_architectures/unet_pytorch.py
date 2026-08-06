import torch.nn as nn
from torch.nn import LeakyReLU, MaxPool2d

""" 
UNet architecture using the classic design but with padding to ensure same size images after convolution.
Uses the PyTorch framework
"""

def double_conv_block(input_channels, output_channels, kernel_size, padding):
    nn.Sequential(
        nn.Conv2d(input_channels, output_channels , kernel_size=3, stride=3),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1, inplace=True),
        nn.Conv2d(output_channels, output_channels , kernel_size=3, stride=3),
        nn.ReLU(inplace=True)
    )
    return nn.Sequential()

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=3)

        self.encoder_conv1 = double_conv_block(1, 64, 3, padding=1)
        self.encoder_conv2 = double_conv_block(64, 128, 3, padding=1)
        self.encoder_conv3 = double_conv_block(128, 256, 3, padding=1)
        self.encoder_conv4 = double_conv_block(256, 512, 3, padding=1)



        self.decoder_conv1 = double_conv_block(512, 256, 3, padding=1)
        self.decoder_conv1 = double_conv_block(256, 128, 3, padding=1)
        self.decoder_conv1 = double_conv_block(128, 64, 3, padding=1)
        self.decoder_conv1 = double_conv_block(64, 256, 3, padding=1)
