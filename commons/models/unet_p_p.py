import torch
import torch.nn as nn
import monai.networks.nets as nets


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetPlusPlus, self).__init__()

        # Define the building blocks of UNet++
        self.encoder1 = nets.UNet(
            dimensions=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
        )

        self.encoder2 = nets.UNet(
            dimensions=3,
            in_channels=out_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
        )

        self.decoder1 = nets.UNet(
            dimensions=3,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            channels=(512, 256, 128, 64),
            strides=(2, 2, 2),
            num_res_units=2,
        )

        self.decoder2 = nets.UNet(
            dimensions=3,
            in_channels=out_channels * 3,
            out_channels=out_channels,
            channels=(256, 128, 64, 32),
            strides=(2, 2, 2),
            num_res_units=2,
        )

        self.final_conv = nn.Conv3d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)

        # Decoder Path
        dec1 = self.decoder1(torch.cat((enc2, enc1), dim=1))
        dec2 = self.decoder2(torch.cat((dec1, enc2, enc1), dim=1))

        # Final Convolution
        out = self.final_conv(torch.cat((dec2, dec1, enc2, enc1), dim=1))

        return out
