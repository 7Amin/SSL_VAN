import time
import warnings

import numpy as np
import torch
import nibabel as nib
import scipy.ndimage as ndi
import torch.nn.parallel
import torch.utils.data.distributed

from torch.cuda.amp import autocast
from utils.utils import AverageMeter, distributed_all_gather
from monai.transforms import Activations, AsDiscrete
from monai.data import decollate_batch


def convert_tensor_to_nii():
    pass


def test_eval(model, loader, acc_func, args, model_inferer=None, post_label=None, post_pred=None, post_post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            print(batch_data)
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
            test_output_image_convert = [post_post_pred(test_pred_tensor) for test_pred_tensor in test_labels_list]
            warnings.warn("test_output_image_convert {}".format(test_output_image_convert[0].shape))
            warnings.warn("test_output_image_convert {}".format(test_output_image_convert[0].max()))
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
            warnings.warn("acc {}".format(np.nan_to_num(acc.cpu().numpy()[0], nan=1.0)))

            if args.rank == 0:
                Dice_background = run_acc.avg[0]
                Dice_spleen = run_acc.avg[1]
                Dice_rkid = run_acc.avg[2]

                Dice_lkid = run_acc.avg[3]
                Dice_gall = run_acc.avg[4]
                Dice_eso = run_acc.avg[5]

                Dice_liver = run_acc.avg[6]
                Dice_sto = run_acc.avg[7]
                Dice_aorta = run_acc.avg[8]

                Dice_IVC = run_acc.avg[9]
                Dice_veins = run_acc.avg[10]
                Dice_pancreas = run_acc.avg[11]

                Dice_rad = run_acc.avg[12]
                Dice_lad = run_acc.avg[13]
                warnings.warn("test {}/{},\n Dice_background: {},\n Dice_spleen: {},\n Dice_rkid: {},\n Dice_lkid: {},\n"
                              " Dice_gall: {},\n Dice_eso: {},\n Dice_liver: {},\n Dice_sto: {},\n Dice_aorta: {},\n"
                              " Dice_IVC: {},\n Dice_veins: {},\n Dice_pancreas: {},\n Dice_rad: {},\n Dice_lad: {}\n"
                              " time {:.2f}s"
                              .format(idx, len(loader), Dice_background, Dice_spleen, Dice_rkid, Dice_lkid, Dice_gall,
                                      Dice_eso, Dice_liver, Dice_sto, Dice_aorta, Dice_IVC, Dice_veins, Dice_pancreas,
                                      Dice_rad, Dice_lad, time.time() - start_time))
                avg_acc = np.mean(run_acc.avg)
                warnings.warn("Test {}/{}  acc {}  time {:.2f}s".format(idx, len(loader),
                                                                        avg_acc, time.time() - start_time))
            start_time = time.time()
            break
    return run_acc.avg


def run_testing(
    model,
    test_loader,
    acc_func,
    args,
    model_inferer=None,
    post_label=None,
    post_pred=None,
):
    epoch_time = time.time()
    # post_sigmoid = Activations(sigmoid=True)
    post_post_pred = AsDiscrete(argmax=True, n_classes=args.out_channels)
    test_avg_acc = test_eval(
        model,
        test_loader,
        acc_func=acc_func,
        args=args,
        model_inferer=model_inferer,
        post_label=post_label,
        post_pred=post_pred,
        post_post_pred=post_post_pred
    )

    test_avg_acc = np.mean(test_avg_acc)

    if args.rank == 0:
        warnings.warn("Final test acc: {:.4f}  time {:.2f}s".format(test_avg_acc, time.time() - epoch_time))

    return test_avg_acc
