import torch
import torch.nn as nn


class NumberEmbedding(nn.Module):
    def __init__(self, dim=128, output_size=600):
        super(NumberEmbedding, self).__init__()
        self.embedding = nn.Embedding(output_size, dim)
        self.linear1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(dim, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        inpt = self.embedding(x)
        embed = self.linear1(inpt)
        res = self.linear2(embed)

        return res, embed
