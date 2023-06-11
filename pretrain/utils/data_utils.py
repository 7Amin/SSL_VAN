from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
import numpy as np
import torch
import json
import math
import os


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


def get_loader(args):
    splits1 = "/dataset_LUNA16_0.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    splits3 = "/dataset_HNSCC_0.json"
    splits4 = "/dataset_TCIAcolon_v2_0.json"
    splits5 = "/dataset_LIDC_0.json"

    list_dir = "../jsons"

    datadir1 = "/media/amin/SP PHD U3/CT_Segmentation_Images/3D/LUNA_16"
    datadir2 = "/media/amin/Amin/CT_Segmentation_Images/3D/TCIAcovid19"
    datadir3 = "/media/amin/Amin/CT_Segmentation_Images/3D/HNSCC"
    datadir4 = "/media/amin/Amin/CT_Segmentation_Images/3D/TCIA_CT_Colonography_Trial"
    datadir5 = "/media/amin/SP PHD U3/CT_Segmentation_Images/3D/TCIA_LIDC"

    if args.mode == "server":
        list_dir = "./jsons"
        datadir1 = "/home/karimimonsefi.1/images/LUNA_16/"
        datadir2 = "/home/karimimonsefi.1/images/TCIAcovid19/"
        datadir3 = "/home/karimimonsefi.1/images/HNSCC/"
        datadir4 = "/home/karimimonsefi.1/images/Colonography/"
        datadir5 = "/home/karimimonsefi.1/images/TCIA_LIDC/"

    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5

    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 LUNA16: OK number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: OK number of data: {}".format(len(datalist2)))
    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print("Dataset 3 HNSCC: OK number of data: {}".format(len(datalist3)))
    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    print("Dataset 4 TCIA Colon: OK number of data: {}".format(len(datalist4)))
    datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    print("Dataset 5 TCIA LIDC: OK number of data: {}".format(len(datalist5)))

    datalist = new_datalist1 + datalist2 + datalist3 + datalist4 + datalist5

    print("Dataset all training: number of data: {}".format(len(datalist)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=1,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    train_ds = Dataset(data=datalist, transform=train_transforms)

    # if args.distributed:
    #     train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    # else:
    #     train_sampler = None
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.workers, sampler=train_sampler, drop_last=True,
        shuffle=(train_sampler is None)
    )
    print("loader is ready")
    return train_loader
