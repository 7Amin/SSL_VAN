import time
import warnings

import numpy as np
import torch
import os
import nibabel as nib
import scipy.ndimage as ndi
import torch.nn.parallel
import torch.utils.data.distributed

from torch.cuda.amp import autocast
from utils.utils import AverageMeter, distributed_all_gather
from monai.transforms import Activations, AsDiscrete
from monai.data import decollate_batch
from monai.metrics import compute_hausdorff_distance


def convert_tensor_to_nii(image_name, args, images):
    for image in images:
        image = torch.squeeze(image).cpu().numpy().astype(np.uint8)
        warnings.warn("image shape{}".format(image.shape))
        nifti_img = nib.Nifti1Image(image, affine=np.eye(4))
        url = os.path.join(args.output_url, image_name)
        nib.save(nifti_img, url)
        warnings.warn("saved at {}".format(url))


def test_eval(model, loader, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    hd95 = AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            val_labels_list = decollate_batch(target)
            logits = torch.nan_to_num(logits, nan=-0.1)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [torch.nan_to_num(post_pred(post_sigmoid(val_pred_tensor)), nan=0.0)
                                  for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                val_output_convert = torch.stack(val_output_convert)
                val_labels_list = torch.stack(val_labels_list)
                val_output_convert = (val_output_convert > 0.5).float()

                hd_distance = compute_hausdorff_distance(val_output_convert,
                                                         val_labels_list,
                                                         percentile=95.0,
                                                         include_background=True)
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                temp = hd95.avg
                warnings.warn("temp {}".format(temp))
                warnings.warn("hd_distance {}".format(hd_distance))
                for i in range(3):
                    if torch.isnan(hd_distance[0][i]) and hd95.count > 0:
                        hd_distance[0][i] = temp[0][i]
                if not torch.isnan(hd_distance).any():
                    hd95.update(hd_distance)

            if args.rank == 0:
                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                warnings.warn("test {}/{}, Dice_TC: {}, Dice_WT: {}, Dice_ET: {}, HD95: {}, time {:.2f}s"
                              .format(idx, len(loader), Dice_TC, Dice_WT, Dice_ET, hd95.avg,
                                      time.time() - start_time))

            start_time = time.time()

    return run_acc.avg, hd95.avg


def run_testing(
    model,
    test_loader,
    acc_func,
    args,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    epoch_time = time.time()
    test_avg_acc, hd95_avg = test_eval(
        model,
        test_loader,
        acc_func=acc_func,
        args=args,
        model_inferer=model_inferer,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )

    test_avg_acc = np.mean(test_avg_acc)

    if args.rank == 0:
        warnings.warn("Final test acc: {:.4f}, hd95_avg {:.4f}, time {:.2f}s".format(test_avg_acc, hd95_avg.mean(),
                                                                                    time.time() - epoch_time))

    return test_avg_acc
