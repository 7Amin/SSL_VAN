import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pretrain.data_loader.luna16_dataset import Luna16Dataset


def get_loader(args, data_type="training"):
    dataset = None
    num_workers = args.num_workers or 4
    if "Luna16Dataset" in args.dataset_name:
        dataset = Luna16Dataset(args, data_type)

    shuffle = False
    if data_type == "training":
        shuffle = True
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers,
                             drop_last=True, shuffle=shuffle)

    return data_loader
