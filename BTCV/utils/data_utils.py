import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist


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
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

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


def get_loader(args):
    data_dir = args.base_data
    datalist_json = args.json_list
    print(datalist_json)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            # add channel to data
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resamples the loaded image and label data to have the specified voxel spacing.
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            # Scales the intensity values of the image data to the specified range. The clip parameter controls
            # whether values outside the input intensity range should be clipped or not
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # Crops the loaded image and label data to the smallest bounding box that contains all foreground
            # voxels in the image data. This is useful for removing empty regions around the edges of the image
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            # Randomly crops a spatial region of size.
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=24,
                image_key="image",
                label_key="label",
                image_threshold=0,
            ),
            # Randomly flips the loaded image and label data along the specified spatial axis with a probability
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            # Randomly rotates the loaded image and label data by a multiple of 90 degrees with a probability
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            # transform is used to randomly shift the intensity values of the image data
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            # transform is used to randomly shift the intensity values of the image data
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            # transform is used to convert the image and label data from numpy arrays to PyTorch tensors
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
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            # todo can remove later
            # Randomly crops a spatial region of size.
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=64,
                image_key="image",
                label_key="label",
                image_threshold=0,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    # val_transform = transforms.Compose(
    #     [
    #         transforms.LoadImaged(keys=["image", "label"]),
    #         transforms.AddChanneld(keys=["image", "label"]),
    #         transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         transforms.Spacingd(
    #             keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
    #         ),
    #         transforms.ScaleIntensityRanged(
    #             keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
    #         ),
    #         transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
    #         transforms.ToTensord(keys=["image", "label"]),
    #     ]
    # )


    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
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
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader
#
#
# class Config:
#     def __init__(self):
#         self.base_data = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training'
#         self.json_list = '../../input_list/dataset_BTCV_List.json'
#         self.space_x = 1.5
#         self.space_y = 1.5
#         self.space_z = 2.0
#         self.a_min = -175.0
#         self.a_max = 250.0
#         self.b_min = 0.0
#         self.b_max = 1.0
#         self.roi_x = 96
#         self.roi_y = 96
#         self.roi_z = 96
#         self.RandFlipd_prob = 0.2
#         self.RandRotate90d_prob = 0.2
#         self.RandScaleIntensityd_prob = 0.1
#         self.RandShiftIntensityd_prob = 0.1
#         self.test_mode = False
#         self.workers = 0
#         self.distributed = False
#
#
# a = get_loader(args=Config())
