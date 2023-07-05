import torch
import torch.nn as nn


class Projection2(nn.Module):
    def __init__(self, input_dim, x_dim, y_dim, z_dim, cluster_num, class_size):
        super(Projection2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_dim, cluster_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(cluster_num),
            nn.GELU(),
            nn.Conv3d(cluster_num, cluster_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(cluster_num),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(y_dim, class_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(class_size),
            nn.GELU(),
            nn.Conv3d(class_size, class_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(class_size),
            nn.GELU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(x_dim, x_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(x_dim // 16),
            nn.GELU(),
            nn.Conv3d(x_dim // 16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(1),
            nn.GELU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(z_dim, z_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(z_dim),
            nn.GELU(),
            nn.Conv3d(z_dim, z_dim, kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x):
        # print("start proj")
        # b, c, x, y, z
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        # x = x.transpose(1, 2).contiguous()
        # x = x.transpose(1, 3).transpose(2, 3).transpose(3, 4).contiguous()
        x = x.permute(0, 3, 2, 1, 4)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.permute(0, 2, 1, 3, 4)
        # x = x.transpose(1, 4).transpose(3, 4).transpose(2, 3).contiguous()
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.permute(0, 4, 3, 2, 1)
        # x = x.transpose(4, 1).transpose(1, 2).transpose(2, 3).contiguous()
        # print(x.shape)
        x = self.conv4(x)
        x = x.squeeze()
        # print(x.shape)
        # print("end proj")
        return x
