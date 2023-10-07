import argparse
import os
import warnings
import numpy as np
import json
import builtins

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from pretrain.utils.data_utils import get_loader
from commons.model_factory import get_pre_trained_model, load_model
from commons.optimizer import get_optimizer
from pretrain.training_2 import run_training_2
from pretrain.losses.loss import ClusteringLoss2


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
    "--cluster_path", default="./cluster_paths.json", type=str, help="path to clusters path :D"
)
parser.add_argument(
    "--pretrained_model_name",
    default="VAN.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
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
parser.add_argument("--test_mode", default=False, type=bool, help="this runner is a test or not")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--num_stages", default=4, type=int, help="number of stages in attention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--embed_dims", default=[64, 128, 256, 512], nargs='+', type=int, help="VAN3D embed dims")
parser.add_argument("--depths", default=[3, 4, 6, 3], nargs='+', type=int, help="VAN3D depths")
parser.add_argument("--mlp_ratios", default=[8, 8, 4, 4], nargs='+', type=int, help="VAN3D mlp_ratios")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--embedding_dim", default=128, type=int, help="embed value for clustering representor")
parser.add_argument("--max_cluster_size", default=512, type=int, help="maximum number of cluster size in clustering")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--logdir", default="./runs/pre_train_1/test_log", type=str,
                    help="directory to save the tensorboard logs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--dist_backend", default="nccl", type=str, help="dist init_process_group backend=nccl")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--optim_name", default="adamw", choices=['adam', 'adamw', 'sgd'],
                    type=str, help="optimization algorithm")
parser.add_argument("--optim_lr", default=1e-2, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--upsample", default="deconv", type=str, choices=['deconv', 'vae'])
parser.add_argument("--model_inferer", default='inferer', type=str, choices=['none', 'inferer'])
parser.add_argument("--valid_loader", default='valid_loader', type=str, choices=['none', 'valid_loader'])
parser.add_argument("--freeze", default="no", type=str, choices=['yes', 'no'], help="freeze pretrain layers in training")
parser.add_argument("--model_v", default='PREVANV6GL', type=str, choices=['PREVANV6GL', 'PREVANV5GL', 'PREVANV412',
                                                                          'PREVANV4GL', 'PREVANV4', 'PREVANV4121GL',
                                                                          'PREVANV4122GL', 'PREVANV4121double'])
parser.add_argument("--mode", default='test', type=str, choices=['server', 'test'])
parser.add_argument("--patch_count", default=2, type=int, help="split image to patches")
parser.add_argument("--mask_length", default=5, type=int, help="an integer value")
parser.add_argument("--phi_1", default=0.1, type=float, help="see paper")
parser.add_argument("--phi_2", default=0.7, type=float, help="see paper")
parser.add_argument("--embed_dim", default=128, type=int, help="output embed dimension")
parser.add_argument("--class_size", default=500, type=int, help="size of class per cluster")
parser.add_argument("--cluster_num", default=1, type=int, help="number of clusters")
parser.add_argument("--apply_mask", default=True, type=bool, help="applying mask for calculating loss")
parser.add_argument("--pretrain_v", default=1, type=int, choices=[1, 2], help="chose version of pretraining")

#  DDP config

parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--dist_backend", default="nccl", type=str, help="dist init_process_group backend=nccl")
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')


def load_train_objects(args):
    loader = get_loader(args)
    model = get_pre_trained_model(args)
    clustering_loss2 = ClusteringLoss2()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters count {pytorch_total_params}")
    start_epoch = 0
    print(f"Total args.checkpoint {args.checkpoint}")
    base_url = 'pre_train_' + \
               '-'.join([str(elem) for elem in args.embed_dims]) + "_" + \
               '-'.join([str(elem) for elem in args.depths]) + "_" + \
               '-'.join([str(elem) for elem in args.mlp_ratios]) + "_" +\
               args.upsample + "_" + args.model_inferer + "_" + args.valid_loader + "_" + args.model_v + \
               "_" + str(args.cluster_num) + "_" + str(args.class_size) + "_" + str(args.embed_dim) + "_" + \
               str(args.apply_mask) + "_" + str(args.mask_length) + "_pre_version" + str(args.pretrain_v)

    args.best_model_url = base_url + "_" + "_best.pt"
    args.final_model_url = base_url + "_" + "_final.pt"
    print(f" Best url model is {args.best_model_url}, final model url is {args.final_model_url}")
    best_loss = 100000000000.0
    optimizer = get_optimizer(model, args)
    # scheduler = get_lr_schedule(args, optimizer, start_epoch)
    scheduler = None
    if args.checkpoint is not None and args.checkpoint:
        model, optimizer, scheduler, best_loss, start_epoch = load_model(args, model, 
                                                                         optimizer, 
                                                                         scheduler, 
                                                                         best_loss,
                                                                         start_epoch)
        print("=> loaded checkpoint '{}' (epoch {}) (best_loss {})".format(
              args.checkpoint, start_epoch, best_loss))
    start_epoch = max(0, start_epoch)
    return loader, model, optimizer, clustering_loss2, scheduler, start_epoch, best_loss



def main():
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    world_size = torch.cuda.device_count()
    print(f'World Size: {world_size}')
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    loader, model, optimizer, clustering_loss2, scheduler, start_epoch, best_loss = load_train_objects(args)
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    torch.backends.cudnn.benchmark = True     
    loss_value = run_training_2(
                                model=model,
                                train_loader=loader,
                                optimizer=optimizer,
                                loss_func=clustering_loss2,
                                args=args,
                                scheduler=scheduler,
                                start_epoch=start_epoch,
                                best_loss=best_loss
                            )
    return loss_value                             


if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
