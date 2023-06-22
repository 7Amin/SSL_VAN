import argparse
import os
import warnings
import numpy as np
from functools import partial

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from BTCV.utils.data_utils import get_loader
from commons.model_factory import get_model, load_model
from commons.pre_trained_loader import load_pre_trained
from commons.optimizer import get_optimizer, get_lr_schedule
from commons.util import fix_outputs_url
from BTCV.trainer import run_training
from BTCV.tester import run_testing

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--checkpoint", action="store_true", help="start training from saved checkpoint")
parser.add_argument("--base_data",
                    default='/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training',
                    type=str, help="base direction of data")
parser.add_argument("--json_list", default='../input_list/dataset_BTCV_List.json',
                    type=str, help="direction of json file of luna16 dataset")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name",
    # default="64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_VANV4GL_2__best.pt",
    default="64-128-256-512_3-4-6-3_8-8-4-4_vae___VANV4__best.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--test_mode", default=False, action="store_true", help="this runner is a test or not")
parser.add_argument("--val_mode", default=False, action="store_true", help="this runner is a validation or not")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--num_stages", default=4, type=int, help="number of stages in attention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--embed_dims", default=[64, 128, 256, 512], nargs='+', type=int, help="VAN3D embed dims")
parser.add_argument("--depths", default=[3, 4, 6, 3], nargs='+', type=int, help="VAN3D depths")
parser.add_argument("--mlp_ratios", default=[8, 8, 4, 4], nargs='+', type=int, help="VAN3D mlp_ratios")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--logdir", default="./runs/BTCV/test_log", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--dist_backend", default="nccl", type=str, help="dist init_process_group backend=nccl")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--upsample", default="vae", type=str, choices=['deconv', 'vae'])
parser.add_argument("--model_inferer", default='inferer', type=str, choices=['none', 'inferer'])
parser.add_argument("--valid_loader", default='valid_loader', type=str, choices=['none', 'valid_loader'])
parser.add_argument("--model_v", default='VANV5GL', type=str, choices=['VAN', 'VANV2', 'VANV3', 'VANV4', 'VANV4GL',
                                                                   'VANV4GLV1', 'VANV4GLV2', 'VANV5GL', "VANV6GL"])
parser.add_argument("--patch_count", default=2, type=int, help="split image to patches")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        warnings.warn("Found total gpus {}".format(args.ngpus_per_node))
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    loader = get_loader(args)
    warnings.warn(f"{args.rank} gpu {args.gpu}")
    if args.rank == 0:
        warnings.warn(f"Batch size is: {args.batch_size} epochs {args.max_epochs}")

    model = get_model(args)

    if args.use_ssl_pretrained:
        try:
            model = load_pre_trained(args, model)
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    #  The Dice coefficient is a metric used to measure the similarity between two sets, and it is often used in image
    #  segmentation tasks in deep learning. By default, the regular Dice coefficient is used, but if this flag is set,
    #  the computation will use the squared version of the Dice coefficient.
    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    #  The AsDiscrete transform is used to convert the output of the segmentation model into discrete labels.
    #  The to_onehot=True parameter specifies that the labels should be converted to a one-hot encoding format,
    #  and the n_classes parameter specifies the number of classes in the segmentation task. This transform is applied
    #  to the ground truth labels (post_label) and the predicted labels (post_pred).
    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    #  The DiceMetric is a metric used to evaluate the similarity between the predicted segmentation and the ground
    #  truth segmentation. The include_background=True parameter includes the background class (i.e., the pixels that
    #  do not belong to any of the segmented classes) in the computation. The reduction=MetricReduction.MEAN parameter
    #  specifies that the mean of the Dice scores should be computed over all samples in the batch. The
    #  get_not_nans=True parameter specifies that the metric should only return a value for batches where the ground
    #  truth labels are not all zeros.
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    if args.model_inferer == 'none':
        model_inferer = None
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    warnings.warn(f"Total parameters count {pytorch_total_params}")

    start_epoch = 0
    warnings.warn(f"Total args.checkpoint {args.checkpoint}")
    base_url = '-'.join([str(elem) for elem in args.embed_dims]) + "_" + \
               '-'.join([str(elem) for elem in args.depths]) + "_" + \
               '-'.join([str(elem) for elem in args.mlp_ratios]) + "_" +\
               args.upsample + "_" + args.model_inferer + "_" + args.valid_loader + "_" + args.model_v

    fix_outputs_url(args, base_url)
    warnings.warn(f" Best url model is {args.best_model_url}, final model url is {args.final_model_url}")
    best_acc = 0.0
    optimizer = get_optimizer(model, args)
    scheduler = get_lr_schedule(args, optimizer, start_epoch)

    if args.checkpoint is not None and args.checkpoint:
        model, optimizer, scheduler, best_acc, start_epoch = load_model(args, model, optimizer, scheduler, best_acc,
                                                                        start_epoch)
        warnings.warn("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(
            args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    if args.test_mode:
        accuracy = run_testing(model=model,
                               val_loader=loader,
                               acc_func=dice_acc,
                               args=args,
                               model_inferer=model_inferer,
                               post_label=post_label,
                               post_pred=post_pred)
    else:
        accuracy = run_training(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            args=args,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
            val_acc_max=best_acc,
        )
    return accuracy


if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    main()
