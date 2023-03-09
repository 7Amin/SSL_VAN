import torch
import torch.nn as nn

from timm.models.layers import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init, trunc_normal_init)
from torch.nn.modules.utils import _pair as to_2tuple

from mmcv.cnn import build_norm_layer
import math
import warnings
from util_models.attentions.LKA3D import SpatialAttention3D


class DWConv3D(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3D, self).__init__()
        self.dwconv3d = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv3d(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv3d = DWConv3D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv3d(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block3D(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        # self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = SpatialAttention3D(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        #  for masking
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                          requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                          requires_grad=True)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(B, C, D, H, W)

        x = self.norm1(x)
        x = self.attn(x)
        x = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x
        x = x + self.drop_path(x)

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x
        x = x + self.drop_path(x)

        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed3D(nn.Module):
    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2, patch_size // 2))
        #  todo remove
        # norm_cfg = dict(type='BN', requires_grad=True)
        # self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        self.norm = nn.BatchNorm3d(embed_dim)

    def forward(self, x):

        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, D, H, W


class VAN3D(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 # num_stages=4,
                 num_stages=1,
                 linear=False,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(VAN3D, self).__init__()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed3D(patch_size=7 if i == 0 else 3,
                                              stride=4 if i == 0 else 2,
                                              in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                              embed_dim=embed_dims[i])

            block = nn.ModuleList([Block3D(dim=embed_dims[i],
                                           mlp_ratio=mlp_ratios[i],
                                           drop=drop_rate,
                                           drop_path=dpr[cur + j],
                                           linear=linear,
                                           norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(VAN3D, self).init_weights()

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, D, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, D, H, W)
            x = norm(x)
            x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
            outs.append(x)

        return outs
