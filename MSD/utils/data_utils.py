import json
import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.transforms import AsDiscreted


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def data_read(datalist, basedir, task="Task01_BrainTumour"):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[task]
    for t in ["training", "validation", 'test']:
        for k in json_data[t]:
            k["image"] = os.path.join(basedir, task, k["image"])
            k["label"] = os.path.join(basedir, task, k["label"])

    tr = json_data['training']
    val = json_data['validation']
    test_urls = json_data['test']
    return tr, val, test_urls


def get_loader(args):
    data_dir = args.base_data
    datalist_json = args.json_list
    train_files, validation_files, test_files = data_read(datalist=datalist_json, basedir=data_dir, task=args.task)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                label_key="label",
                image_threshold=0,
            ),
            # AsDiscreted(keys=["label"], threshold_values=True, logit_thresh=0.5, n_classes=args.out_channels),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.valid_loader == 'none':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                AsDiscreted(keys=["label"], threshold_values=True, logit_thresh=0.5, n_classes=args.out_channels),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    label_key="label",
                    image_threshold=0,
                ),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

    train_ds = None
    val_ds = None
    if args.test_mode:
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        val_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True
        )
        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader, train_ds, val_ds
