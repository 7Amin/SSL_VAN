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
from monai.transforms import Randomizable
from typing import Optional
import random
import numpy as np
import torch
import os


class RandomSelect(Randomizable):
    def __init__(self, prob: float, percent: Optional[float] = None):
        self.prob = prob
        self.percent = percent

    def randomize(self, data):
        self._do_transform = self.R.random() < self.prob
        if self._do_transform and self.percent is not None:
            data_shape = data["image"].shape
            self.indices = np.random.choice(data_shape[0], int(data_shape[0] * self.percent), replace=False)

    def __call__(self, data):
        if self._do_transform and self.percent is not None:
            data["image"] = data["image"][self.indices]
        return data


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
            # RandomSelect(prob=1.0, percent=0.1),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.num_samples,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    train_ds = Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, drop_last=True
        , shuffle=True
    )
    print("loader is ready")
    return train_loader
