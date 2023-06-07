import os
import time
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from pretrain.utils.utils import load_clusters


from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, clusters):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, _ = batch_data
        else:
            data, _ = batch_data["image"], batch_data["label"]
        data = data.cuda(args.rank)
        data_numpy = data.numpy()
        merged_array = np.reshape(data_numpy,
                                  (data_numpy.shape[0] * data_numpy.shape[1],
                                   data_numpy.shape[2] * data_numpy.shape[3]))
        target = np.zeros((data_numpy.shape[0], data_numpy.shape[1], len(clusters)))
        for index, cluster in enumerate(clusters):
            temp = cluster.predict(merged_array)
            temp = temp.reshape((data_numpy.shape[0], data_numpy.shape[1]))
            target[:, :, index] = temp

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            #
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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

        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


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
    optimizer,
    loss_func,
    args,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            warnings.warn(f"Writing Tensorboard logs to {args.logdir}")
    scaler = None
    if args.amp:
        scaler = GradScaler()
    train_loss = None
    clusters = load_clusters(args.cluster_path)
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        warnings.warn(f"{args.rank}  {time.ctime()}  Epoch: {epoch}")
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func,
            args=args, clusters=clusters
        )
        if args.rank == 0:
            warnings.warn("Final training  {}/{}  loss: {:.4f}  time {:.2f}s".format(epoch, args.max_epochs - 1,
                                                                                     train_loss,
                                                                                     time.time() - epoch_time))
            save_checkpoint(
                model, epoch, args, best_acc=train_loss, optimizer=optimizer, scheduler=scheduler,
                filename=args.final_model_url
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        if scheduler is not None:
            scheduler.step()

    return train_loss
