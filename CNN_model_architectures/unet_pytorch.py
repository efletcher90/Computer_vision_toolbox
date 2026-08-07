import torch.nn as nn
import torch
import config

""" 
UNet architecture using the classic design but with padding to ensure same size input and output images.
Uses the PyTorch framework
"""

# ---------- UNet encoder and decoder block architecture ---------- #

class DoubleConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1, inplace=True),
            nn.Conv2d(output_channels, output_channels , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, data):
        return self.double_conv(data)

class DownSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.convolution = DoubleConvBlock(input_channels, output_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, data):
        encoded_data = self.convolution(data) # this is separate to the pool variable below as this will be concatenated following the skip
        pooled_data = self.pooling(encoded_data) # halve spatial resolution through pooling

        return encoded_data, pooled_data

class UpSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, out_channels=input_channels//2 , kernel_size=2, stride=2)
        self.convolution = DoubleConvBlock(input_channels, output_channels)

    def forward(self, data, data_skip):
        decoded_data = self.up(data)
        data_up = torch.cat([decoded_data, data_skip], dim=1) # dim=1 refers to the channel axis

        return self.convolution(data_up)

# ---------- loss functions for binary segmentation ---------- #

class DiceLoss(nn.Module):
    """
    Standalone Dice loss, operating on raw logits (sigmoid applied
    internally to logit results). This is equivalent to my previous use of Keras' Dice (scores: 0 = perfect, 1 = worst)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        probs = nn.Sigmoid(logits)
        probs = probs.view(-1)
        labels = labels.view(-1)
        intersection = (probs * labels).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs.sum() + labels.sum() + + self.smooth)
        return 1 - dice_coeff

class BCEDiceLoss(nn.Module):
    """
    Hybrid loss function combining binary cross entropy and dice loss.
    The idea is that it takes both the pixel level classification from the BCE and the
    intersection-over-union (IOU) score from Dice
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, labels):
        return self.bce(logits, labels) + self.dice(logits, labels)

# ---------- UNet architecture ---------- #

""" 
Differences between PyTorch and Keras

PyTorch expects the input tensor shape  as (batch, channel, height, width) e.g. (100 images, 1 for GS, x=1024px, y=1024px).
Keras uses (batch, height, width, channel) e.g. (100 images, x=1024px, y=1024px, 1 for GS).

Keras has activation function (i.e. sigmoid for binary usage) and loss function written explicitly during model compile. 
PyTorch 
"""

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()

        self.encoder_conv1 = DownSample(input_channels, 64)
        self.encoder_conv2 = DownSample(64, 128)
        self.encoder_conv3 = DownSample(128, 256)
        self.encoder_conv4 = DownSample(256, 512)

        self.bottle_neck = DoubleConvBlock(512, 1024)

        self.decoder_conv1 = UpSample(1024, 512)
        self.decoder_conv2 = UpSample(512, 256)
        self.decoder_conv3 = UpSample(256, 128)
        self.decoder_conv4 = UpSample(128, 64)

        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):
        down_1, pool_1 = self.encoder_conv1(inputs)
        down_2, pool_2 = self.encoder_conv2(pool_1)
        down_3, pool_3 = self.encoder_conv3(pool_2)
        down_4, pool_4 = self.encoder_conv4(pool_3)

        b_neck = self.bottle_neck(pool_4)

        up_1 = self.decoder_conv1(b_neck, down_4)
        up_2 = self.decoder_conv2(up_1, down_3)
        up_3 = self.decoder_conv3(up_2, down_2)
        up_4 = self.decoder_conv4(up_3, down_1)

        output = self.output(up_4)

        return output

if __name__ == "__main__":
    def build_UNet():
        model = UNet(input_channels=1, num_classes=1)
        loss_f = config.LOSS_F
        optimiser = torch.optim.Adam(model.parameters(), lr=config.L_RATE)
