#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Partial Source: https://github.com/espnet/espnet/blob/master/espnet2/hubert/hubert_loss.py

import torch.nn.functional as F
from torch import nn


class HubertPretrainLoss(nn.Module):
    """Hubert criterion module.

    Args:
        pred_masked_weight: weight for predictive loss for masked frames
        pred_nomask_weight: weight for predictive loss for unmasked frames
        loss_weights: weights for additional loss terms (not first one)
    """

    def __init__(
        self,
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
        loss_weights: float = 10.0,
    ):
        super(HubertPretrainLoss, self).__init__()

        # in the paper these two values are alpha and 1 - alpha
        # respectively. The default is alpha = 1, which means completely using
        # loss over the masked items.

        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights

    def forward(self, logp_m_list, targ_m_list, logp_u_list, targ_u_list, reduce=True):
        """
        logp_m_list: List of logits for predicted labels on masked sequence elements
        targ_m_list: List of target classifications from cluster ensemble
        logp_u_list: Ditto for the unmasked elements of the sequence
        targ_u_list: Ditto for the unmasked elements of the sequence
        """
        loss = 0.0
        sample_size = 0
        reduction = "sum" if reduce else "none"

        if self.pred_masked_weight > 0:
            loss_m_list = []
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                loss_m_list.append(loss_m)

            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()

        if self.pred_nomask_weight > 0:
            loss_u_list = []
            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
                loss_u_list.append(loss_u)

            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()

        # if self.loss_weights > 0:
        #     assert hasattr(model, "get_extra_losses")
        #     extra_losses, names = model.get_extra_losses(enc_outputs)

        #     if isinstance(extra_losses, list):
        #         extra_losses = extra_losses[0]
        #         names = names[0]
        #     else:
        #         raise NotImplementedError("only support one extra loss")
        #     loss += self.loss_weights * extra_losses.float() * sample_size

        # could return logp_m_list, logp_u_list to calculate accuracy

        return loss
