import torch
import torch.nn as nn

from monai.utils import ensure_tuple_rep
from util_models.van_3d import VAN3D


class VAN(nn.Module):
    def __init__(self, embed_dims, mlp_ratios, depths, num_stages, in_channels, out_channels, dropout_path_rate,
                 upsample="vae", project_dim=512, image_size=96*96, cluster_number=400, max_cluster_size=500):
        super(VAN, self).__init__()
        self.van3d = VAN3D(in_chans=in_channels, drop_path_rate=dropout_path_rate, embed_dims=embed_dims,
                           mlp_ratios=mlp_ratios, depths=depths, num_stages=num_stages)

        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(embed_dims[-1], 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(embed_dims[-1], 512)
        self.projection = nn.Linear(image_size * out_channels, project_dim * cluster_number * max_cluster_size)
        self.project_dim = project_dim
        self.cluster_number = cluster_number
        self.max_cluster_size = max_cluster_size
        if upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(embed_dims[-1], embed_dims[-1] // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 2, embed_dims[-1] // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 4, embed_dims[-1] // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 8, embed_dims[-1] // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(embed_dims[-1] // 16, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
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
                nn.Conv3d(embed_dims[-1] // 16, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        b, seq, c, h, w = x.shape
        # x = x.view(b, c, seq, h, w)
        x = self.van3d(x.contiguous())[-1]
        # x = x.flatten(start_dim=2, end_dim=4)
        x = self.conv(x)
        # x = x.squeeze(1)
        y = x.view(b, seq, c * h * w)
        y = self.projection(y)
        y = y.view(b, seq, self.cluster_number, self.max_cluster_size, self.project_dim)
        return x, y
