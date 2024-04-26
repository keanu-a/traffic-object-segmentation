import torch
import torch.nn as nn

PADDING = 1
STRIDE = 2
DOUBLE_CONV_STRIDE = 1
DOUBLE_CONV_KERNEL_SIZE = 3
SAMPLE_KERNEL_SIZE = 2


# Added padded because I want input to be same size as the output
class DoubleConv(nn.Module):
    """
        Two 3x3 convolutions, each followed by a rectified linear unit (ReLU)
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=DOUBLE_CONV_KERNEL_SIZE, padding=PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=DOUBLE_CONV_KERNEL_SIZE, padding=PADDING),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class DownSample(nn.Module):
    """
        Used for encoding
        2x2 max pooling operation with stride 2 for downsampling
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=SAMPLE_KERNEL_SIZE, stride=STRIDE)

    def forward(self, x):
        down_sample = self.conv(x)
        max_pool = self.pool(down_sample)

        return down_sample, max_pool
    

class UpSample(nn.Module):
    """
        Used for decoding
        Feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels
        A concatenation with the correspondingly cropped feature map from the contracting path
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=SAMPLE_KERNEL_SIZE, stride=STRIDE)
        self.conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)