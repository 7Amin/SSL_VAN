import os
import shutil
import time
import json
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed


from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

def replace_nan(tensor, replace_value):
    tensor[torch.isnan(tensor)] = replace_value
    return tensor


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        # warnings.warn("target {}".format(target.shape))
        # warnings.warn("data {}".format(data.shape))
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        # target = target * 1.0
        # target_num_classes = torch.argmax(target, dim=1, keepdim=True)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            # warnings.warn("target {}".format(target.shape))
            logits = model(data)
            # warnings.warn("logits {}".format(logits.shape))
            # loss = loss_func(logits, target_num_classes)
            # loss = (loss_func(logits[:, 0, :, :, :], target[:, 0, :, :, :]) + \
            #        loss_func(logits[:, 1, :, :, :], target[:, 2, :, :, :]) + \
            #        loss_func(logits[:, 2, :, :, :], target[:, 1, :, :, :])) / 10.1

            loss = loss_func(logits.double(), target.double())

        # if torch.isnan(loss):
        #     loss = prev_loss
        #     warnings.warn(
        #         "Epoch {}/{} {}/{} NAN LOSS time {:.2f}s".format(epoch, args.max_epochs, idx, len(loader),
        #                                                                time.time() - start_time))
        if args.rank == 0:
            warnings.warn(f"loss is {loss} and {type(loss)} and {torch.isnan(loss)}")
        if torch.isnan(loss):
            continue
        if args.amp:
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            warnings.warn(
                "Epoch {}/{} {}/{}  loss: {:.4f}  time {:.2f}s".format(epoch, args.max_epochs, idx, len(loader),
                                                                 run_loss.avg, time.time() - start_time))
        # else:
        #     break
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            val_labels_list = decollate_batch(target.double())
            val_outputs_list = decollate_batch(logits.double())
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            # if args.rank == 0:
            #     warnings.warn(f"value of output is {val_output_convert[0]}")
            #     warnings.warn(f"max value of output is {max(val_output_convert)}")

            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            if args.rank == 0:
                warnings.warn(f"acc is f{acc}")
                warnings.warn(f"not_nans is f{not_nans}")
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
                # avg_acc_1 = np.mean(run_acc)
                # warnings.warn("Val {}/{} {}/{}  acc {}  time {:.2f}s".format(epoch, args.max_epochs, idx, len(loader),
                #                                                              avg_acc_1, time.time() - start_time))
                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                warnings.warn("Val {}/{} {}/{}, Dice_TC: {}, Dice_WT: {},"
                              " Dice_ET: {}, time {:.2f}s".format(epoch, args.max_epochs, idx, len(loader), Dice_TC,
                                                                  Dice_WT, Dice_ET, time.time() - start_time))

            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model_final.pt", best_acc=0.0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    warnings.warn(f"Saving checkpoint {filename}")


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
    val_acc_max=0.0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            warnings.warn(f"Writing Tensorboard logs to {args.logdir}")
    scaler = None
    if args.amp:
        scaler = GradScaler()
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        warnings.warn(f"{args.rank}  {time.ctime()}  Epoch: {epoch}")
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            warnings.warn("Final training  {}/{}  loss: {:.4f}  time {:.2f}s".format(epoch, args.max_epochs - 1,
                                                                                     train_loss,
                                                                                     time.time() - epoch_time))
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            # val_acc = np.mean(val_acc)
            if args.rank == 0:
                warnings.warn("Final validation  {}/{}  acc: {}  time {:.2f}s".format(epoch, args.max_epochs - 1,
                                                                                          val_acc,
                                                                                          time.time() - epoch_time))
                Dice_TC = val_acc[0]
                Dice_WT = val_acc[1]
                Dice_ET = val_acc[2]
                warnings.warn("Final validation {}/{}, Dice_TC: {}, Dice_WT: {},"
                              " Dice_ET: {}, time {:.2f}s".format(epoch, args.max_epochs, Dice_TC,
                                                                  Dice_WT, Dice_ET, time.time() - epoch_time))
                # val_avg_acc = val_acc
                val_avg_acc = np.mean(val_acc)
                warnings.warn("Final validation ({:.6f})".format(val_avg_acc))
                if val_avg_acc > val_acc_max:
                    warnings.warn("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler,
                            filename=args.final_model_url
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler,
                                filename=args.final_model_url)
                if b_new_best:
                    warnings.warn("Copying to best model new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, args.final_model_url),
                                    os.path.join(args.logdir, args.best_model_url))

        if scheduler is not None:
            scheduler.step()

    warnings.warn("Training Finished !, Best Accuracy: {}".format(val_acc_max))

    return val_acc_max
