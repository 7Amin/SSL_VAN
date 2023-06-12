import time
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from torch.cuda.amp import autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def test_eval(model, loader, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            test_labels_list = decollate_batch(target)
            test_labels_convert = [post_label(test_label_tensor) for test_label_tensor in test_labels_list]
            test_labels_list = decollate_batch(logits)
            test_output_convert = [post_pred(test_pred_tensor) for test_pred_tensor in test_labels_list]
            acc_func.reset()
            acc_func(y_pred=test_output_convert, y=test_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                warnings.warn("Test {}/{}  acc {}  time {:.2f}s".format(idx, len(loader),
                                                                        avg_acc, time.time() - start_time))
            start_time = time.time()
    return run_acc.avg


def run_testing(
    model,
    val_loader,
    acc_func,
    args,
    model_inferer=None,
    post_label=None,
    post_pred=None,
):
    epoch_time = time.time()
    val_avg_acc = test_eval(
        model,
        val_loader,
        acc_func=acc_func,
        model_inferer=model_inferer,
        args=args,
        post_label=post_label,
        post_pred=post_pred,
    )

    val_avg_acc = np.mean(val_avg_acc)

    if args.rank == 0:
        warnings.warn("Final validation acc: {:.4f}  time {:.2f}s".format(val_avg_acc, time.time() - epoch_time))

    return val_avg_acc
