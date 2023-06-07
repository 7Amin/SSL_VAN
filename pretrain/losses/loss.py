import torch
from torch import nn
import numpy as np


class IoU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def __call__(self, predictions, targets):
        return get_IoU(predictions, targets, self.args) * -1.0


def get_IoU(predictions, targets, args):
    predictions = predictions > 0.5
    targets = targets > 0.5

    # Calculate intersection and union
    intersection = torch.logical_and(predictions, targets).float().sum()
    union = torch.logical_or(predictions, targets).float().sum()

    # Calculate IoU, handling the case where the union is 0
    iou = torch.where(union > 0.0, intersection / union, torch.tensor(0.0).to(args.device))

    return iou


class ClusteringLoss(nn.Module):
    def __init__(self, embedding_dim, max_cluster_size):
        super(ClusteringLoss, self).__init__()
        self.max_cluster_size = max_cluster_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_cluster_size, embedding_dim)

    def forward(self, outputs, targets):
        b, seq, cluster_number = targets.shape
        targets_new = targets.unsqueeze(-1)
        targets_new = targets_new.view(b * seq * cluster_number, 1)
        targets_new = self.embedding(targets_new)
        targets_new = targets_new.view(b, seq, cluster_number, self.embedding_dim)

        cos_sim = torch.nn.functional.cosine_similarity(outputs, targets_new.unsqueeze(3), dim=-1)

        # apply softmax along the embedding dimension
        softmax_cos_sim = torch.nn.functional.softmax(cos_sim, dim=-1)
        # res = softmax_cos_sim[b, seq, cluster_number, targets]
        res = softmax_cos_sim[np.arange(b)[:, np.newaxis, np.newaxis],
                              np.arange(seq)[:, np.newaxis],
                              np.arange(cluster_number),
                              targets]
        res = torch.log(res)
        res = res.sum(dim=(-2, -1)) / (seq * cluster_number * -1.0)

        return res


def get_loss(loss_name, args):
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss().to(args.device)
    if loss_name == 'IoU':
        return IoU(args).to(args.device)
    if loss_name == 'L1Loss':
        return torch.nn.L1Loss().to(args.device)
    if loss_name == 'MSELoss':
        return torch.nn.MSELoss().to(args.device)


# def test():
#     embedding_dim = 5
#     max_cluster_size = 15
#     b = 4
#     seq = 96
#     cluster_number = 25
#     loss_func = ClusteringLoss(embedding_dim, max_cluster_size)
#     output_matrix = torch.randn(b, seq, cluster_number, max_cluster_size, embedding_dim)
#     target_matrix = torch.randint(0, max_cluster_size, (b, seq, cluster_number))
#     loss_value = loss_func(output_matrix, target_matrix)
#     print(loss_value)
#
#
# test()
