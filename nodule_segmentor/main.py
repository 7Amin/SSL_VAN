import argparse
import os
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


from nodule_segmentor.data_loader.data_utils import get_loader
from nodule_segmentor.model.van import VAN
from nodule_segmentor.losses.loss import Loss, get_IoU
from nodule_segmentor.optimizers.lr_scheduler import WarmupCosineSchedule


parser = argparse.ArgumentParser(description="PyTorch Training")

parser.add_argument("--num_workers", default=0, type=int, help="number of worker for loading data")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--size_x", default=128, type=int, help="size image for x")
parser.add_argument("--size_y", default=128, type=int, help="size image for y")
parser.add_argument("--patch_size", default=96, type=int, help="patch size")
parser.add_argument("--base_data", default="/media/amin/SP PHD U3/CT_Segmentation_Images/3D",
                    type=str, help="base direction of data")
parser.add_argument("--luna_data", default="/LUNA_16/manifest-1600709154662", type=str,
                    help="direction of Luna16 data")
parser.add_argument("--base_dir_code", default="/home/amin/CETI/medical_image/SSL_VAN/",
                    type=str, help="direction of start point of code")
parser.add_argument("--luna16_json", default="input_list/dataset_LUNA16_List.json",
                    type=str, help="direction of json file of luna16 dataset")
parser.add_argument("--dataset_name", default="-Luna16Dataset-",
                    type=str, help="names of all datasets")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument("--logdir", default="test_log", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
parser.add_argument("--multi_gpu", default=False, type=bool, help="using single gpu or multi gpu")
parser.add_argument("--num_stages", default=1, type=int, help="number of stages in attention")
parser.add_argument("--embed_dims", default=[64], nargs='+', type=int, help="VAN3D embed dims")
parser.add_argument("--depths", default=[3], nargs='+', type=int, help="VAN3D depths")
parser.add_argument("--mlp_ratios", default=[8], nargs='+', type=int, help="VAN3D mlp_ratios")
parser.add_argument("--grad_clip", default=True, action="store_true", help="gradient clip")
parser.add_argument("--lrdecay", default=True, action="store_true", help="enable learning rate decay")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")


parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
# parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")


args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.multi_gpu:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args.device = "cpu"

args.distributed = False

training_data_loader, validation_data_loader = get_loader(args)

model = VAN(args, upsample="deconv")
model = model.to(args.device)

loss_function = Loss(args)
optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

logdir = "./runs/" + args.logdir
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

with open('./runs/model.txt', 'w') as f:
    f.write(str(model))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Number of trainable parameters: {count_parameters(model)}")


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def train(args, global_step, train_loader, val_best):
    model.train()
    loss_train = []
    loss = None
    for data in train_loader:
        t1 = time()
        x, y = data
        x = x.to(args.device)
        y = y.to(args.device)
        predicted = model(x)
        loss = loss_function(predicted, y)
        loss_train.append(loss.item())
        loss.backward()

        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if args.lrdecay:
            scheduler.step()
        optimizer.zero_grad()
        iou = get_IoU(x, predicted)
        print("Step:{}/{}, Loss:{:.6f}, Time:{:.4f}, IoU:{:.4f}".format(global_step,
                                                                        args.num_steps, loss, time() - t1, iou))

        global_step += 1

        val_cond = global_step % args.eval_num == 0

        if val_cond:
            val_loss = validation(validation_data_loader)
            if val_loss < val_best:
                val_best = val_loss
                checkpoint = {
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                print(
                    "Model was saved ! Best Recon. Val Loss: {:.4f}".format(
                        val_best
                    )
                )
            else:
                print(
                    "Model was not saved ! Best Recon. Val Loss: {:.4f}".format(
                        val_best
                    )
                )
    return global_step, loss, val_best


def validation(test_loader):
    model.eval()
    loss_val = []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            predicted = model(x)
            loss = loss_function(predicted, y)
            loss_val.append(loss.item())
            iou = get_IoU(x, predicted)
            print("Validation step:{}, Loss:{:.4f}, IoU: {:.4f}".format(step, loss, iou))

    return np.mean(loss_val)


global_step = 0
val_best = np.inf
for i in range(args.epochs):
    global_step, loss, val_best = train(args, global_step, training_data_loader, val_best)
    checkpoint = {"epoch": i, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss}
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


torch.save(model.state_dict(), logdir + "final_model.pth")
