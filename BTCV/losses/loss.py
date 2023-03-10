import torch


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


def get_loss(loss_name, args):
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss().to(args.device)
    if loss_name == 'IoU':
        return IoU(args).to(args.device)
    if loss_name == 'L1Loss':
        return torch.nn.L1Loss().to(args.device)
    if loss_name == 'MSELoss':
        return torch.nn.MSELoss().to(args.device)
