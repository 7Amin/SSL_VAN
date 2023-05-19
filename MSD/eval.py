import os
import time
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import autocast


from MSD.utils.data_utils import get_loader
from MSD.model.van import VAN
from utils.utils import AverageMeter
from utils.plot_images import plot_ans_save_segmentation, sample_stack

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch


def val_epoch(model, loader, acc_func, args, post_label=None, post_pred=None):
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
                logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            plot_ans_save_segmentation(val_output_convert, val_labels_convert, data, idx)
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                warnings.warn("Val {}/{}  acc {}  time {:.2f}s".format(idx, len(loader),
                                                                       avg_acc, time.time() - start_time))
            start_time = time.time()
    return run_acc.avg


def main_worker(gpu, args):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    loader = get_loader(args)
    model = VAN(embed_dims=args.embed_dims,
                mlp_ratios=args.mlp_ratios,
                depths=args.depths,
                num_stages=args.num_stages,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                dropout_path_rate=args.dropout_path_rate,
                upsample=args.upsample)

    base_url = '-'.join([str(elem) for elem in args.embed_dims]) + "_" + \
               '-'.join([str(elem) for elem in args.depths]) + "_" + \
               '-'.join([str(elem) for elem in args.mlp_ratios]) + "_" +\
               args.upsample
    args.best_model_url = base_url + "_" + "_best.pt"
    model_dict = torch.load(os.path.join("../", args.logdir, args.best_model_url))["state_dict"]
    model.load_state_dict(model_dict)
    warnings.warn("Use model_final weights")

    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)

    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    warnings.warn(f"Total parameters count {pytorch_total_params}")

    model.cuda(args.gpu)

    val_loader = loader[1]
    val_avg_acc = val_epoch(
        model,
        val_loader,
        acc_func=dice_acc,
        args=args,
        post_label=post_label,
        post_pred=post_pred,
    )
    return val_avg_acc


class Config:
    def __init__(self):
        self.base_data = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/MSD/Abdomen/RawData/Training'
        self.json_list = '../input_list/dataset_MSD_List.json'
        self.space_x = 1.5
        self.space_y = 1.5
        self.space_z = 2.0
        self.a_min = -175.0
        self.a_max = 250.0
        self.b_min = 0.0
        self.b_max = 1.0
        self.roi_x = 96
        self.roi_y = 96
        self.roi_z = 96
        self.batch_size = 1
        self.workers = 1
        self.RandFlipd_prob = 0.2
        self.RandRotate90d_prob = 0.2
        self.RandScaleIntensityd_prob = 0.1
        self.RandShiftIntensityd_prob = 0.1
        self.test_mode = False
        self.distributed = False
        self.logdir = './runs/MSD/test_log'
        self.gpu = 0
        self.rank = 0
        self.embed_dims = [64, 128, 256, 512]
        self.mlp_ratios = [8, 8, 4, 4]
        self.depths = [3, 4, 6, 3]
        self.num_stages = 4
        self.in_channels = 1
        self.out_channels = 14
        self.dropout_path_rate = 0.0
        self.use_normal_dataset = True
        self.amp = False
        self.upsample = "deconv"


acc = main_worker(gpu=0, args=Config())
