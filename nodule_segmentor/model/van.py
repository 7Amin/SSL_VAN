import torch
import torch.nn as nn

from monai.utils import ensure_tuple_rep
from util_models.van_3d import VAN3D


class VAN(nn.Module):
    def __init__(self, args, upsample="vae"):
        super(VAN, self).__init__()
        self.args = args
        embed_dims = args.embed_dims
        mlp_ratios = args.mlp_ratios
        depths = args.depths
        self.van3d = VAN3D(in_chans=args.in_channels, drop_path_rate=args.dropout_path_rate,
                           embed_dims=embed_dims, mlp_ratios=mlp_ratios, depths=depths, num_stages=args.num_stages)
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(embed_dims[-1], 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(embed_dims[-1], 512)
        if upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(embed_dims[-1], embed_dims[-1] // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 2, embed_dims[-1] // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 4, embed_dims[-1] // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 8, embed_dims[-1] // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 16, embed_dims[-1] // 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 32, embed_dims[-1] // 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 64, embed_dims[-1] // 128, kernel_size=(2, 2, 2),
                                   stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 128, embed_dims[-1] // 256, kernel_size=(2, 2, 2),
                                   stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 256, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(embed_dims[-1], embed_dims[-1] // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 2, embed_dims[-1] // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 4, embed_dims[-1] // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 8, embed_dims[-1] // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 16, embed_dims[-1] // 32, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 32),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 32, embed_dims[-1] // 64, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 64),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 64, embed_dims[-1] // 128, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 128),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 128, embed_dims[-1] // 256, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(embed_dims[-1] // 256),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(embed_dims[-1] // 256, args.in_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x = self.van3d(x.contiguous())[-1]
        x = x.flatten(start_dim=2, end_dim=4)
        x = self.conv(x)
        return x
