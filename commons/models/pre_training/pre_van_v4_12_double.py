import torch
import torch.nn as nn

from commons.models.util_models.van_3d import VAN3D
from commons.models.pre_training.pretrain_projection_2 import Projection2


class ConvBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super(ConvBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        if output_dim is None:
            output_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim),
            # nn.SyncBatchNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1),
            # nn.SyncBatchNorm(output_dim),
            nn.BatchNorm3d(output_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class PREVANV4121double(nn.Module):
    def __init__(self, embed_dims, mlp_ratios, depths, num_stages, in_channels, out_channels, dropout_path_rate,
                 upsample="deconv", cluster_num=10, class_size=10, x_dim=96, y_dim=96,z_dim=96, args=None):
        super(PREVANV4121double, self).__init__()
        self.van3d = VAN3D(in_chans=in_channels, drop_path_rate=dropout_path_rate, embed_dims=embed_dims,
                           mlp_ratios=mlp_ratios, depths=depths, num_stages=num_stages)

        self.num_stages = num_stages
        self.embed_dims = [in_channels, embed_dims[0]] + embed_dims

        for i in range(num_stages):
            conv_res = ConvBlock(self.embed_dims[-1 - i])
            setattr(self, f"conv_res{i}", conv_res)

        self.first_conv = ConvBlock(self.embed_dims[0], self.embed_dims[1])
        self.final_conv0 = ConvBlock(self.embed_dims[1])
        self.pre_train_proj = Projection2(input_dim=self.embed_dims[1], x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                                          cluster_num=cluster_num, class_size=class_size)

        for i in range(num_stages):
            upsample = nn.Sequential(
                ConvBlock(self.embed_dims[-1 - i], self.embed_dims[-2 - i]),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            setattr(self, f"upsample{i}", upsample)
        upsample = nn.Sequential(
            ConvBlock(self.embed_dims[1]),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
        setattr(self, f"final_upsample", upsample)

    def forward(self, x):
        first_x = x
        first_x = self.first_conv(first_x)
        van_res = self.van3d(x.contiguous())

        for i in range(self.num_stages):
            v = van_res[-i - 1]
            conv_res = getattr(self, f"conv_res{i}")
            v = conv_res(v)
            if i != 0:
                # x = torch.cat((x, v), dim=1)
                x = (x + v) / 2.0
            else:
                x = v.contiguous()
            upsample = getattr(self, f"upsample{i}")
            x = upsample(x)            

        final_upsample = getattr(self, "final_upsample")
        x = self.final_conv0(x)
        x = final_upsample(x)
        # x = torch.cat((first_x, x), dim=1)
        x = (x + first_x) / 2.0
        x = self.pre_train_proj(x)

        return x
