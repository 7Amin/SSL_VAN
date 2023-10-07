import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from pretrain.utils.utils import load_clusters
from pretrain.utils.mask import apply_mask



class Trainer:
    def _get_target(self, data):
        data_numpy = data.detach().numpy()
        b, z, x, y = data.shape
        merged_array = np.reshape(data_numpy, (b * z, x * y)).astype(np.double)

        target = np.zeros((b, z, self.args.cluster_num))
        for index, cluster in enumerate(self.clusters):
            if index >= self.args.cluster_num:
                break
            temp = cluster.predict(merged_array)
            temp = temp.reshape((b, z))
            for i in range(b):
                for j in range(z):
                    target[i, j, index] = temp[i][j]

        return torch.from_numpy(target).double()
    
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler, train_loader, best_loss, start_epoch, 
                 loss_func, args) -> None:
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_loader
        self.args = args
        self.loss_func = loss_func
        self.start_epoch = start_epoch
        self.model = model
        self.best_loss = best_loss

    def _save_checkpoint(self, epoch=0, average_loss=0.0):
        state_dict = self.model.state_dict() if not self.args.distributed else self.model.module.state_dict()
        save_dict = {"epoch": epoch, "best_acc": average_loss, "state_dict": state_dict}
        if self.optimizer is not None:
            save_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()
        filename = os.path.join(self.args.logdir, self.args.final_model_url)
        torch.save(save_dict, filename)
        print(f"Saving checkpoint {filename}")

    def _run_batch(self, inputs):
        data = inputs["image"]
        data = data.squeeze(dim=1)
        target = self._get_target(data)
        data = data.unsqueeze(1)
        data, mask = apply_mask(data, self.args)
        data, target = inputs["image"], inputs["label"]
        data, target = data.cuda(self.args.gpu), target.cuda(self.args.gpu)
        
        self.optimizer.zero_grad()
        logits = self.model(data)

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

    def train(self):
        self.clusters = load_clusters(self.args.cluster_path)

        for epoch in range(self.start_epoch, self.args.max_epochs):
            epoch_time = time.time()
            self.model.train()
            total_loss = self._run_epoch(epoch)
            average_loss = total_loss / len(self.train_data)
            print(f"Epochs is {epoch} | Loss is {average_loss} | time {time.time() - epoch_time}s")
            if (epoch + 1) % self.args.val_every == 0 and self.args.rank == 0 and self.args.logdir is not None and self.args.checkpoint:
                self._save_checkpoint(epoch=epoch, average_loss=average_loss)
                print("Copying to best model new best model!!!!")
                shutil.copyfile(os.path.join(self.args.logdir, self.args.final_model_url),
                                os.path.join(self.args.logdir, self.args.best_model_url))
                
        print("Training Finished !, Best Accuracy: {}".format(self.val_acc_max))
        return self.val_acc_max
