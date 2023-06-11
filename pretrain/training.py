import os
import time
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from pretrain.utils.utils import load_clusters
from pretrain.utils.mask import apply_mask


from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather


def get_target(data, clusters, embed_dim, embed_number_values):
    data_numpy = data.detach().numpy()
    b, z, x, y = data.shape
    merged_array = np.reshape(data_numpy, (b * z, x * y)).astype(np.double)
    print(merged_array.dtype)
    target = np.zeros((b, z, len(clusters), embed_dim))
    print(embed_number_values)
    for index, cluster in enumerate(clusters):
        print(cluster)
        print(index)
        temp = cluster.predict(merged_array)
        temp = temp.reshape((b, z))
        warnings.warn("temp shape".format(temp.shape))
        for i in range(b):
            for j in range(z):
                key_ = int(temp[i][j])
                print(key_)
                embed_value = embed_number_values[key_]
                target[i, j, index] = embed_value

    return target


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, clusters, embed_dim, embed_number_values):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data = batch_data
        else:
            data = batch_data["image"]

        print(data.shape)
        # print(data.dtype)
        # data = data.float()
        data = data.squeeze()
        print(data.shape)
        # print(data.dtype)
        target = get_target(data, clusters, embed_dim, embed_number_values)
        data = data.cuda(args.rank)
        target = target.cuda(args.rank)
        data = data.unsqueeze(1)
        print(data.shape)
        print(target.shape)
        data, mask = apply_mask(data, args)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(outputs=logits, targets=target, mask=mask, apply_mask=args.apply_mask)
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
    embed_dim=512,
    embed_number_values=None
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
            args=args, clusters=clusters, embed_dim=embed_dim, embed_number_values=embed_number_values
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
