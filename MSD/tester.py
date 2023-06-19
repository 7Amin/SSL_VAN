import time
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from torch.cuda.amp import autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def test_eval(model, loader, args, model_inferer=None, post_pred=None):
    model.eval()
    start_time = time.time()
    res = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, _ = batch_data
            else:
                data = batch_data["image"]
            data = data.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            print(logits.shape)
            test_labels_list = decollate_batch(logits)
            test_output_convert = [post_pred(test_pred_tensor) for test_pred_tensor in test_labels_list]
            res.append(test_output_convert)
            if args.rank == 0:
                warnings.warn("Test {}/{}  time {:.2f}s".format(idx, len(loader), time.time() - start_time))
            start_time = time.time()
    return res


def run_testing(
    model,
    val_loader,
    args,
    model_inferer=None,
    post_pred=None,
):
    epoch_time = time.time()
    res = test_eval(
        model,
        val_loader,
        args=args,
        model_inferer=model_inferer,
        post_pred=post_pred,
    )
    if args.rank == 0:
        warnings.warn("Final test len: {}  time {:.2f}s".format(len(res), time.time() - epoch_time))

    return res
