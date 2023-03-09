import torch


class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.recon_loss = torch.nn.MSELoss().to(args.device)

    def __call__(self, output_recons, target_recons):
        # _, d, _, _ = target_recons.shape
        print(target_recons.shape)
        # recon_loss = self.recon_loss(output_recons[:, : d, :, :], target_recons)
        recon_loss = self.recon_loss(output_recons, target_recons)

        return recon_loss
