import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_loader(args):
    splits1 = "/dataset_LUNA16_0.json"
    # # splits2 = "/dataset_TCIAcovid19_0.json"
    # splits3 = "/dataset_HNSCC_0.json"
    # splits4 = "/dataset_TCIAcolon_v2_0.json"
    # splits5 = "/dataset_LIDC_0.json"
    num_workers = 4
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 LUNA16: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)

    if args.distributed:
        train_sampler = sampler = DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())

    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader
