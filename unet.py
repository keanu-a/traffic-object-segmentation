import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample


# Using UNet due to simple architecture
class UNet(nn.Module):
    def __init__(self, in_channels, class_amount):
        super(UNet, self).__init__()

        self.encoder_1 = DownSample(in_channels, 32)
        self.encoder_2 = DownSample(32, 64)
        self.encoder_3 = DownSample(64, 128)
        self.encoder_4 = DownSample(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.decoder_1 = UpSample(512, 256)
        self.decoder_2 = UpSample(256, 128)
        self.decoder_3 = UpSample(128, 64)
        self.decoder_4 = UpSample(64, 32)

        self.output = nn.Conv2d(in_channels=32, out_channels=class_amount, kernel_size=1)

    def forward(self, x):
        # ENCODING
        e_1, mp_1 = self.encoder_1(x)
        e_2, mp_2 = self.encoder_2(mp_1)
        e_3, mp_3 = self.encoder_3(mp_2)
        e_4, mp_4 = self.encoder_4(mp_3)

        # BOTTLENECK
        bn = self.bottleneck(mp_4)

        # DECODING
        d_1 = self.decoder_1(bn, e_4)
        d_2 = self.decoder_2(d_1, e_3)
        d_3 = self.decoder_3(d_2, e_2)
        d_4 = self.decoder_4(d_3, e_1)

        out = self.output(d_4)
        return out