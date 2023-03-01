import argparse
import os
import torch
from pretrain.data_loader.data_utils import get_loader


parser = argparse.ArgumentParser(description="PyTorch Training")

parser.add_argument("--num_workers", default=4, type=int, help="number of worker for loading data")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--size_x", default=512, type=int, help="size image for x")
parser.add_argument("--size_y", default=512, type=int, help="size image for y")
parser.add_argument("--base_data", default="/media/amin/SP PHD U3/CT_Segmentation_Images/3D",
                    type=str, help="base direction of data")
parser.add_argument("--luna_data", default="/LUNA_16/manifest-1600709154662", type=str,
                    help="direction of Luna16 data")
parser.add_argument("--base_dir_code", default="/home/amin/CETI/medical_image/SSL_VAN",
                    type=str, help="direction of start point of code")
parser.add_argument("--luna16_json", default="/input_list/dataset_LUNA16_List.json",
                    type=str, help="direction of json file of luna16 dataset")
parser.add_argument("--dataset_name", default="-Luna16Dataset-",
                    type=str, help="names of all datasets")



parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
# parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.distributed = False

training_data_loader = get_loader(args, "training")
validation_data_loader = get_loader(args, "validation")
