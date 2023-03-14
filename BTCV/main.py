import argparse
import os
import numpy as np
from functools import partial

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from BTCV.utils.data_utils import get_loader
from BTCV.model.van import VAN
from BTCV.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from BTCV.trainer import run_training

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--base_data",
                    default='/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training',
                    type=str, help="base direction of data")
parser.add_argument("--json_list", default='./input_list/dataset_BTCV_List.json',
                    type=str, help="direction of json file of luna16 dataset")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name",
    default="VAN.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--test_mode", default=False, type=bool, help="this runner is a test or not")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--num_stages", default=1, type=int, help="number of stages in attention")
parser.add_argument("--embed_dims", default=[64], nargs='+', type=int, help="VAN3D embed dims")
parser.add_argument("--depths", default=[3], nargs='+', type=int, help="VAN3D depths")
parser.add_argument("--mlp_ratios", default=[8], nargs='+', type=int, help="VAN3D mlp_ratios")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--logdir", default="./run/BTCV/test_log", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
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


def main():
    args = parser.parse_args()
    #  The first argument, --noamp, is a flag that can be used to disable the use of Automatic Mixed Precision (AMP)
    #  during training. AMP is a technique used in deep learning models to improve performance by using lower-precision
    #  data types (e.g. float16) for some parts of the computation. By default, this argument is not set, so AMP will
    #  be used during training unless the user explicitly sets the --noamp flag.
    args.amp = not args.noamp
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
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
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    # todo should remove this
    pretrained_dir = args.pretrained_dir
    model = VAN(embed_dims=args.embed_dims,
                mlp_ratios=args.mlp_ratios,
                depths=args.depths,
                num_stages=args.num_stages,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                dropout_path_rate=args.dropout_path_rate,
                upsample="deconv")

    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    if args.use_ssl_pretrained:
        try:
            model_dict = torch.load("./pretrained_models/model_van.pt")
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised Swin UNETR backbone weights !")
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

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    # The argument --warmup_epochs is an integer that represents the number of warm-up epochs used in training a
    # machine learning model. During warm-up, the learning rate is gradually increased from a very small value to
    # its full value over a certain number of epochs. This helps to prevent the model from getting stuck in local
    # minima and can improve convergence speed and stability.
    # 1 - Set the initial learning rate to a small value, say 0.001.
    # 2 - Gradually increase the learning rate over the first 10 epochs to its full value, say 0.01.
    # 3 - Continue training with a constant learning rate of 0.01 for the remaining epochs.
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=None,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
