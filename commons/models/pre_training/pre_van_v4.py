import torch
import torch.nn as nn

from commons.models.util_models.van_3d import VAN3D
from commons.models.pre_training.pretrain_projection import Projection
from commons.models.pre_training.pretrain_projection_2 import Projection2


class PREVANV4(nn.Module):
    def __init__(self, embed_dims, mlp_ratios, depths, num_stages, in_channels, out_channels, dropout_path_rate,
                 upsample="deconv", cluster_num=400, class_size=600, embed_dim=256, x_dim=96, y_dim=96, z_dim=96,
                 args=None):
        super(PREVANV4, self).__init__()
        self.van3d = VAN3D(in_chans=in_channels, drop_path_rate=dropout_path_rate, embed_dims=embed_dims,
                           mlp_ratios=mlp_ratios, depths=depths, num_stages=num_stages)

        self.num_stages = num_stages

        for i in range(num_stages - 1):
            conv = nn.Sequential(
                nn.Conv3d(embed_dims[-1] // 2 ** i, embed_dims[-1] // 2 ** (i + 1),
                          kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 2 ** (i + 1)),
                nn.LeakyReLU(),
                nn.Conv3d(embed_dims[-1] // 2 ** (i + 1), embed_dims[-1] // 2 ** (i + 1),
                          kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 2 ** (i + 1)),
                nn.LeakyReLU(),
                )
            setattr(self, f"conv{i}", conv)

        self.first_conv = nn.Sequential(
            nn.Conv3d(in_channels, embed_dims[-1] // 2 ** num_stages,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(embed_dims[-1] // 2 ** num_stages),
            nn.LeakyReLU(),
            nn.Conv3d(embed_dims[-1] // 2 ** num_stages, embed_dims[-1] // 2 ** (num_stages - 1),
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(embed_dims[-1] // 2 ** (num_stages - 1)),
            nn.LeakyReLU()
            )

        self.final_conv0 = nn.Sequential(
                nn.Conv3d(embed_dims[-1] // 2 ** num_stages, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(),
                )
        if args.pretrain_v == 1:
            self.pre_train_proj = Projection(input_dim=out_channels + embed_dims[-1] // 2 ** (num_stages - 1),
                                             x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                                             cluster_num=cluster_num, class_size=class_size, embed_dim=embed_dim)
        else:
            self.pre_train_proj = Projection2(input_dim=out_channels + embed_dims[-1] // 2 ** (num_stages - 1),
                                              x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                                              cluster_num=cluster_num, class_size=class_size)

        if upsample == "deconv":
            for i in range(num_stages):
                upsample = nn.ConvTranspose3d(embed_dims[-1] // 2**i, embed_dims[-1] // 2**(i + 1),
                                              kernel_size=(2, 2, 2), stride=(2, 2, 2))
                setattr(self, f"upsample{i}", upsample)
            upsample = nn.ConvTranspose3d(out_channels, out_channels,
                                          kernel_size=(2, 2, 2), stride=(2, 2, 2))
            setattr(self, f"final_upsample", upsample)
        elif upsample == "vae":
            for i in range(num_stages):
                upsample = nn.Sequential(
                    nn.Conv3d(embed_dims[-1] // 2**i, embed_dims[-1] // 2**(i + 1), kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm3d(embed_dims[-1] // 2**(i + 1)),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
                setattr(self, f"upsample{i}", upsample)
            upsample = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            setattr(self, f"final_upsample", upsample)

    def forward(self, x):
        first_x = x
        first_x = self.first_conv(first_x)

        van_res = self.van3d(x.contiguous())

        upsample = getattr(self, f"upsample{0}")
        x = upsample(van_res[-1])

        for i in range(self.num_stages - 1):  # 0, 1, 2
            v = van_res[-i - 2]
            x = torch.cat((x, v), dim=1)
            conv = getattr(self, f"conv{i}")
            x = conv(x)
            upsample = getattr(self, f"upsample{i + 1}")
            x = upsample(x)

        final_upsample = getattr(self, "final_upsample")
        x = self.final_conv0(x)
        x = final_upsample(x)
        x = torch.cat((first_x, x), dim=1)
        x = self.pre_train_proj(x)

        return x
