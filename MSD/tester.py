import time
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from torch.cuda.amp import autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete
from monai.metrics import compute_hausdorff_distance


def test_eval(model, loader, acc_func, args, model_inferer=None, post_label=None, post_pred=None, post_post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    hd95 = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            image_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            # print(image_name)
            warnings.warn("image_name {}".format(image_name))
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
            acc = acc_func(y_pred=test_output_convert, y=test_labels_convert)
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list = distributed_all_gather(
                    [acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list):
                    run_acc.update(al)
            else:
                run_acc.update(np.nan_to_num(acc.cpu().numpy()[0], nan=1.0))
                test_output_convert = torch.stack(test_output_convert)
                test_labels_list = torch.stack(test_labels_list)
                # test_output_convert = (test_output_convert > 0.5).float()
                hd_distance = compute_hausdorff_distance(test_output_convert,
                                                         test_labels_list,
                                                         percentile=95.0,
                                                         include_background=True)
                temp = hd95.avg
                warnings.warn("temp {}".format(temp))
                warnings.warn("hd_distance {}".format(hd_distance))
                for i in range(3):
                    if torch.isnan(hd_distance[0][i]) and hd95.count > 0:
                        hd_distance[0][i] = temp[0][i]
                # if not torch.isnan(hd_distance).any():
                #     hd95.update(hd_distance)
                hd95.update(hd_distance)

            if args.rank == 0:
                # warnings.warn("run_acc.avg {}".format(run_acc.avg))
                list_size = len(run_acc.avg)
                warnings.warn("test {}/{}, len is {}".format(idx, len(loader), list_size))
                for i in range(list_size):
                    warnings.warn("{}: {},".format(i, run_acc.avg[i]))
                warnings.warn(", HD95: {}, time {:.2f}s".format(hd95.avg, time.time() - start_time))

            start_time = time.time()

    return run_acc.avg, hd95.avg


def run_testing(
    model,
    test_loader,
    acc_func,
    args,
    model_inferer=None,
    post_label=None,
    post_pred=None
):
    epoch_time = time.time()
    post_post_pred = AsDiscrete(argmax=True, n_classes=args.out_channels)
    test_avg_acc, hd95_avg = test_eval(
        model,
        test_loader,
        acc_func=acc_func,
        args=args,
        model_inferer=model_inferer,
        post_label=post_label,
        post_pred=post_pred,
        post_post_pred=post_post_pred
    )

    test_avg_acc_mean = np.mean(test_avg_acc)

    if args.rank == 0:
        warnings.warn("Final test acc: {}, acc mean: {:.4f}, hd95_avg {:.4f}, time {:.2f}s".format(
            test_avg_acc, test_avg_acc_mean, hd95_avg.mean(), time.time() - epoch_time))

    return test_avg_acc
