import torch


class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.recon_loss = torch.nn.MSELoss().to(args.device)
        self.recon_loss = torch.nn.CrossEntropyLoss().to(args.device)

    def __call__(self, output_recons, target_recons):
        # _, d, _, _ = target_recons.shape
        print(target_recons.shape)
        # recon_loss = self.recon_loss(output_recons[:, : d, :, :], target_recons)
        recon_loss = self.recon_loss(output_recons, target_recons)

        return recon_loss


class IoU(torch.nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def __call__(self, predictions, targets):
        return get_IoU(predictions, targets) * -1.0


def get_IoU(predictions, targets):
    predictions = predictions > 0.5
    targets = targets > 0.5

    # Calculate intersection and union
    intersection = torch.logical_and(predictions, targets).float().sum()
    union = torch.logical_or(predictions, targets).float().sum()

    # Calculate IoU, handling the case where the union is 0
    iou = torch.where(union > 0.0, intersection / union, torch.tensor(0.0))

    return iou
