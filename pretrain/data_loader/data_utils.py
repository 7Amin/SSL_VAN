import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pretrain.data_loader.luna16_dataset import Luna16Dataset


def get_loader(args, data_type="training"):
    dataset = None
    num_workers = args.num_workers or 4
    if args.dataset_name.contain("Luna16Dataset"):
        dataset = Luna16Dataset(args, data_type)
    if args.distributed and data_type == "training":
        train_sampler = DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(),
                                           rank=torch.distributed.get_rank())
    else:
        train_sampler = None

    shuffle = False
    if data_type == "training":
        shuffle = True

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers,
                             sampler=train_sampler, drop_last=True, shuffle=shuffle)

    return data_loader
