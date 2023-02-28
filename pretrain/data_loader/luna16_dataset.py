import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pydicom import dcmread
from pydicom.data import get_testdata_file


class Luna16Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        # filename = get_testdata_file("MR_small.dcm")
        # ds = dcmread(filename)
        # ds.pixel_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
