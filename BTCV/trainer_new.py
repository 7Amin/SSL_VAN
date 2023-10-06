import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
#  TensorboardX is a library for visualizing and analyzing the training process of deep learning models using
#  TensorBoard, a visualization tool that comes with TensorFlow and Pytorch. TensorboardX allows users to visualize and
#  compare metrics such as loss, accuracy, and gradients, as well as view histograms of weights and biases in real-time
#  during training

from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch



class Trainer: 
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler, train_loader, val_loader, start_epoch, post_label, post_pred,
                 loss_func, acc_func, model_inferer, val_acc_max, args) -> None:
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_loader
        self.val_loader = val_loader
        self.args = args
        self.loss_func = loss_func
        self.start_epoch = start_epoch
        self.model = model
        self.acc_func = acc_func
        self.post_label = post_label 
        self.post_pred = post_pred
        self.model_inferer = model_inferer
        self.val_acc_max = val_acc_max

    def _save_checkpoint(model, epoch, args, filename="model_final.pt", best_acc=0.0, optimizer=None, scheduler=None):
        state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        filename = os.path.join(args.logdir, filename)
        torch.save(save_dict, filename)
        print(f"Saving checkpoint {filename}")

    def _run_batch(self, inputs):
        data, target = inputs["image"], inputs["label"]
        data, target = data.cuda(self.args.gpu), target.cuda(self.args.gpu)
        
        self.optimizer.zero_grad()
        logits = self.model(data)
        # print(f"data shape is {data.shape} | target shape is {target.shape} | predict shape is {logits.shape}")

        loss = self.loss_func(logits, target)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()
    
    def _run_epoch(self, epoch):
        print(f"[GPU {self.args.rank}] Epochs {epoch}, BatchSize: {self.args.batch_size} | Steps: {len(self.train_data)}")
        total_loss = 0
        for idx, batch in enumerate(self.train_data):  
            loss = self._run_batch(batch)
            total_loss += loss
            print(f"[GPU {self.args.rank}] Step/Steps: {idx}/{len(self.train_data)} | Total Loss: {total_loss/(idx + 1.0)} | Loss: {loss}")

        return total_loss
    
    def _run_val(self, epoch):
        print("Start Val")
        self.model.eval()
        run_acc = AverageMeter()
        start_time = time.time()
        with torch.no_grad():
            for idx, batch_data in enumerate(self.val_loader):
                data, target = batch_data["image"], batch_data["label"]
                data, target = data.cuda(self.args.gpu), target.cuda(self.args.gpu)
                if self.model_inferer is not None:
                    logits = self.model_inferer(data)
                else:
                    logits = self.model(data)
                val_labels_list = decollate_batch(target)
                val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(logits)
                val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                self.acc_func.reset()
                self.acc_func(y_pred=val_output_convert, y=val_labels_convert)
                acc, not_nans = self.acc_func.aggregate()
                acc = acc.cuda(self.args.gpu)
                if self.args.distributed:
                    acc_list, not_nans_list = distributed_all_gather(
                        [acc, not_nans], out_numpy=True, is_valid=idx < self.val_loader.sampler.valid_length
                    )
                    for al, nl in zip(acc_list, not_nans_list):
                        run_acc.update(al, n=nl)
                else:
                    run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            avg_acc = np.mean(run_acc.avg)
            # if self.args.rank == 0:
            print("Val {}/{} {}/{}  acc {} | {}  time {:.2f}s".format(epoch, self.args.max_epochs, idx,
                                                                                    len(self.val_loader), run_acc.avg,
                                                                                    avg_acc, time.time() - start_time))
            start_time = time.time()
        return run_acc.avg

    def train(self):
        for epoch in range(self.start_epoch, self.args.max_epochs):
            self.model.train()
            total_loss = self._run_epoch(epoch)
            average_loss = total_loss / len(self.train_data)
            print(f"Epochs is {epoch} | Loss is {average_loss}")

            if (epoch + 1) % self.args.val_every == 0:
                can_replace_best = False
                epoch_time = time.time()
                val_avg_acc = self._run_val(epoch)

                val_avg_acc = np.mean(val_avg_acc)

                if self.args.rank == 0:
                    print("Final validation  {}/{}  acc: {:.4f}  time {:.2f}s".format(epoch, self.args.max_epochs - 1,
                                                                                          val_avg_acc,
                                                                                          time.time() - epoch_time))
                if val_avg_acc > self.val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(self.val_acc_max, val_avg_acc))
                    self.val_acc_max = val_avg_acc
                    can_replace_best = True
                if self.args.rank == 0 and self.args.logdir is not None and self.args.checkpoint:
                    print(f"Saving model in the {self.args.final_model_url}")
                    self._save_checkpoint(
                        model=self.model, epoch=epoch, args=self.args, filename=self.args.final_model_url,
                        best_acc=self.val_acc_max, optimizer=self.optimizer, scheduler=self.scheduler
                    )
                    if can_replace_best:
                        print("Copying to best model new best model!!!!")
                        shutil.copyfile(os.path.join(self.args.logdir, self.args.final_model_url),
                                        os.path.join(self.args.logdir, self.args.best_model_url))
                        
        print("Training Finished !, Best Accuracy: {}".format(self.val_acc_max))
        return self.val_acc_max
