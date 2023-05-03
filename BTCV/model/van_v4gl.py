import torch
import torch.nn as nn

from BTCV.model.van_v4 import VANV4


class VANV4GL(nn.Module):
    def __init__(self, embed_dims, mlp_ratios, depths, num_stages, in_channels, out_channels, dropout_path_rate,
                 upsample="deconv", patch_count=2):
        super(VANV4GL, self).__init__()
        self.patch_count = patch_count
        for i in range(patch_count):
            for j in range(patch_count):
                for k in range(patch_count):
                    setattr(self, f"van{i}_{j}_{k}", VANV4(embed_dims[:-2], mlp_ratios[:-2], depths[:-2],
                                                           num_stages - 2, in_channels, out_channels,
                                                           dropout_path_rate, upsample))
        self.van = VANV4(embed_dims, mlp_ratios, depths, num_stages, in_channels,
                         out_channels, dropout_path_rate, upsample)
        self.conv = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1)
                )

    def forward(self, x):
        #  x is b, c, seq, w, h
        x1 = self.van(x)

        t0s = torch.split(x, self.patch_count, dim=2)
        res_t0s = None
        for i, t0 in enumerate(t0s):
            t1s = torch.split(t0, self.patch_count, dim=3)
            res_t1s = None
            for j, t1 in enumerate(t1s):
                t2s = torch.split(t1, self.patch_count, dim=4)
                res_t2s = None
                for k, t2 in enumerate(t2s):
                    model = getattr(self, f"van{i}_{j}_{k}")
                    print(t2.shape)
                    t2 = model(t2)
                    if k == 0:
                        res_t2s = t2
                    else:
                        res_t2s = torch.cat((res_t2s, t2), dim=4)
                if j == 0:
                    res_t1s = res_t2s
                else:
                    res_t1s = torch.cat((res_t1s, res_t2s), dim=3)

            if i == 0:
                res_t0s = res_t1s
            else:
                res_t0s = torch.cat((res_t0s, res_t1s), dim=3)

        x = x1 + res_t0s
        x = self.conv(x)
        return x
