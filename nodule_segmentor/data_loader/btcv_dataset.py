import monai
import json
import os
import numpy as np
import cv2
import nibabel as nib
import torch
from torch.utils.data import Dataset

from pydicom import dcmread
url = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/averaged-training-images/DET0000101_avg.nii.gz'
url_l = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/averaged-training-labels/DET0000101_avg_seg.nii.gz'
nifti_image = nib.load(url)
nifti_image_l = nib.load(url_l)

# Get the image data as a numpy array
image_array = nifti_image.get_fdata()
image_array_l = nifti_image_l.get_fdata()

# Convert the numpy array to a PyTorch tensor
image_tensor = monai.utils.numpymodule.as_tensor(image_array)
# tensor = monai.data.NibabelReader()
print('a')


class BTCV(Dataset):
    def __init__(self, args, data_type="training", min_val=np.inf, max_val=-np.inf):
        self.args = args
        # self.subjects_info = _read_images_directories(args, data_type)
        # min_val, max_val = get_min_max(args, self.subjects_info, min_val, max_val)
        self.min_val = -3000
        self.max_val = 5000

    def __len__(self):
        return len(self.subjects_info)

    def set_max(self, max_val):
        self.max_val = max_val

    def set_min(self, min_val):
        self.min_val = min_val

    def __getitem__(self, index):
        # subject_info = self.subjects_info[index]

        # data, label = _load_images(self.args, subject_info, self.min_val, self.max_val)
        data, label = None, None
        # print(subject_info)
        # print(data.shape)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()