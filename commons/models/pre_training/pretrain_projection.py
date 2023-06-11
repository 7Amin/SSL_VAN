import torch
import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, input_dim, x_dim, y_dim, z_dim, cluster_num, class_size, embed_dim):
        super(Projection, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_dim, cluster_num, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(cluster_num),
            nn.LeakyReLU())

        self.conv2 = nn.Sequential(
            nn.Conv3d(x_dim, class_size, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(class_size),
            nn.LeakyReLU())

        self.conv3 = nn.Sequential(
            nn.Conv3d(y_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(embed_dim),
            nn.LeakyReLU())

        self.conv4 = nn.Conv3d(z_dim, z_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # b, c, z, x, y
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = x.view(0, 3, 2, 1, 4)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = x.view(0, 4, 2, 3, 1)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = x.view(0, 2, 3, 4, 1)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        return x
