import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self):
        super(ClusteringLoss, self).__init__()

    def forward(self, outputs, targets, mask, apply_mask=True):
        tai = 0.1
        outputs = outputs.double()
        targets = targets.double()
        target_shape = targets.shape  # (b, seq, cluster_number, embedding_dim)
        pred_shape = outputs.shape  # (b, seq, cluster_number, max_cluster_size, embedding_dim)

        targets_reshaped = targets.view(-1, target_shape[-1])  # (b * seq * cluster_number, embedding_dim)
        # (b * seq * cluster_number, max_cluster_size, embedding_dim)
        outputs_reshaped = outputs.view(-1, pred_shape[-2], pred_shape[-1])

        # Calculate cosine similarity
        similarity = F.cosine_similarity(targets_reshaped.unsqueeze(1), outputs_reshaped, dim=-1)
        similarity = similarity.view(*target_shape[:-1], pred_shape[-2])  # (b, seq, cluster_number, max_cluster_size)

        similarity = similarity * tai
        # apply softmax along the embedding dimension
        probabilities = F.softmax(similarity, dim=-1)
        # print(probabilities.shape)

        if apply_mask:
            expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).bool()
            probabilities = torch.masked_select(probabilities, expanded_mask)
        # print(probabilities.shape)
        # Calculate negative log-likelihood loss
        # loss = -torch.log(probabilities + 1e-12).mean()
        loss = -torch.log(probabilities).mean()
        loss.requires_grad_(True)

        return loss


def get_loss(loss_name, args):
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss().to(args.device)
    if loss_name == 'IoU':
        return IoU(args).to(args.device)
    if loss_name == 'L1Loss':
        return torch.nn.L1Loss().to(args.device)
    if loss_name == 'MSELoss':
        return torch.nn.MSELoss().to(args.device)
    if loss_name == "ClusteringLoss":
        return ClusteringLoss()


# def test(apply_mask=True):
#     embedding_dim = 256
#     max_cluster_size = 500
#     b = 1
#     seq = 96
#     cluster_number = 20
#     loss_func = ClusteringLoss()
#     target_matrix = torch.randn(b, seq, cluster_number, embedding_dim)
#     output_matrix = torch.randn(b, seq, cluster_number, max_cluster_size, embedding_dim)
#     mask_matrix = torch.randint(0, 2, size=(b, seq))
#     loss_value = loss_func(output_matrix, target_matrix, mask_matrix, apply_mask)
#     print('loss: {:.9f}'.format(loss_value))
#
#
# i = 0
# apply_mask = True
# while i < 20:
#     print(apply_mask)
#     test(apply_mask)
#     # apply_mask = not apply_mask
#     i = i + 1
