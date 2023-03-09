import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from nodule_segmentor.data_loader.luna16_dataset import Luna16Dataset
from random import randint


def random_patch_collate(batch, patch_size):
    patches = []
    for vol in batch:
        # Choose a random patch from the volume
        x = randint(0, vol.shape[2] - patch_size)
        y = randint(0, vol.shape[3] - patch_size)
        z = randint(0, vol.shape[4] - patch_size)
        patch = vol[:, :, x:x+patch_size, y:y+patch_size, z:z+patch_size]
        patches.append(patch)

    # Stack the patches into a batch tensor
    batch_tensor = torch.stack(patches)

    return batch_tensor


def get_loader(args, data_type="training"):
    dataset = None
    num_workers = args.num_workers or 4
    if "Luna16Dataset" in args.dataset_name:
        dataset = Luna16Dataset(args, data_type)

    shuffle = False
    if data_type == "training":
        shuffle = True
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, drop_last=True,
                             shuffle=shuffle, collate_fn=lambda x: random_patch_collate(x, patch_size=args.patch_size))

    return data_loader
