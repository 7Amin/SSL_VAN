import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from nodule_segmentor.data_loader.luna16_dataset import Luna16Dataset
from random import randint


def random_patch_collate(batch, patch_size):
    patches_x, patches_y = [], []
    for vol_x, vol_y in batch:
        # Choose a random patch from the volume
        x = randint(0, vol_x.shape[1] - patch_size)
        y = randint(0, vol_x.shape[2] - patch_size)
        z = randint(0, vol_x.shape[3] - patch_size)
        patch_x = vol_x[:, :, x:x+patch_size, y:y+patch_size, z:z+patch_size]
        patch_y = vol_y[:, :, x:x+patch_size, y:y+patch_size, z:z+patch_size]
        patches_x.append(patch_x)
        patches_y.append(patch_y)

    # Stack the patches into batch tensors
    batch_tensor_x = torch.stack(patches_x)
    batch_tensor_y = torch.stack(patches_y)

    return batch_tensor_x, batch_tensor_y


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
