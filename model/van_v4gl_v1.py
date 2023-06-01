import torch
import torch.nn as nn

from model.van_v4 import VANV4


class VANV4GLV1(nn.Module):
    def __init__(self, embed_dims, mlp_ratios, depths, num_stages, in_channels, out_channels, dropout_path_rate,
                 upsample="deconv", patch_count=2):
        super(VANV4GLV1, self).__init__()
        self.patch_count = patch_count
        for i in range(patch_count):
            for j in range(patch_count):
                for k in range(patch_count):
                    setattr(self, f"van{i}_{j}_{k}", VANV4(embed_dims[:-1], mlp_ratios[:-1], depths[:-1],
                                                           num_stages - 1, in_channels, out_channels,
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
        split_size_d2 = x.size(2) // self.patch_count
        t0s = torch.split(x, split_size_d2, dim=2)
        res_t0s = None
        # print(f"t0s len is {len(t0s)}")
        for i, t0 in enumerate(t0s):
            # print(f"t0.shape is {t0.shape}")
            split_size_d3 = x.size(3) // self.patch_count
            t1s = torch.split(t0, split_size_d3, dim=3)
            res_t1s = None
            # print(f"t1s len is {len(t1s)}")
            for j, t1 in enumerate(t1s):
                # print(f"t1.shape is {t1.shape}")
                split_size_d4 = x.size(4) // self.patch_count
                t2s = torch.split(t1, split_size_d4, dim=4)
                res_t2s = None
                # print(f"t2s len is {len(t2s)}")
                for k, t2 in enumerate(t2s):
                    # print(f"t2.shape is {t2.shape}")
                    model = getattr(self, f"van{i}_{j}_{k}")
                    t2 = model(t2)
                    if k == 0:
                        res_t2s = t2
                    else:
                        res_t2s = torch.cat((res_t2s, t2), dim=4)
                if j == 0:
                    res_t1s = res_t2s
                    res_t2s = None
                else:
                    res_t1s = torch.cat((res_t1s, res_t2s), dim=3)
                    res_t2s = None

            if i == 0:
                res_t0s = res_t1s
                res_t1s = None
            else:
                res_t0s = torch.cat((res_t0s, res_t1s), dim=2)
                res_t1s = None

        # print(f"res_t0s.shape is {res_t0s.shape}")
        x = self.van(x) + res_t0s
        x = self.conv(x)
        # print(f"x.shape is {x.shape}")
        return x
