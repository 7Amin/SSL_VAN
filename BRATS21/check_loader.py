import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from BRATS21.utils.data_utils import get_loader

class Config:
    def __init__(self):
        self.base_data = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BraTS21'
        self.json_list = '../input_list/dataset_BRATS21_List.json'
        self.space_x = 1.0
        self.space_y = 1.0
        self.space_z = 1.0
        self.a_min = -175.0
        self.a_max = 250.0
        self.b_min = 0.0
        self.b_max = 1.0
        self.roi_x = 128
        self.roi_y = 128
        self.roi_z = 128
        self.batch_size = 2
        self.workers = 1
        self.RandFlipd_prob = 0.5
        self.RandRotate90d_prob = 0.2
        self.RandScaleIntensityd_prob = 0.2
        self.RandShiftIntensityd_prob = 0.2
        self.test_mode = False
        self.distributed = False
        self.logdir = './runs/BraTS/test_log'
        self.gpu = 0
        self.rank = 0
        self.embed_dims = [64, 128, 256, 512]
        self.mlp_ratios = [8, 8, 4, 4]
        self.depths = [3, 4, 6, 3]
        self.num_stages = 4
        self.in_channels = 1
        self.out_channels = 14
        self.dropout_path_rate = 0.0
        self.use_normal_dataset = True
        self.amp = False
        self.upsample = "deconv"
        self.fold = 0
        self.valid_loader = "noasdne"


args = Config()
_, train_ds, val_ds = get_loader(args)
data = train_ds[120]
# data = val_ds[120]
# data = val_ds
slice_id = 73
num = 0
# pick one image from DecathlonDataset to visualize and check the 4 channels
print(f"image shape: {data[num]['image'].shape}")
plt.figure("image", (24, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(data[num]["image"][i, :, :, slice_id].detach().cpu() / 20.0,  cmap="gray")  #
plt.show()
# also visualize the 3 channels label corresponding to this image
print(f"label shape: {data[num]['label'].shape}")
plt.figure("label", (24, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"label channel {i}")
    plt.imshow(data[num]["label"][i, :, :, slice_id].detach().cpu())
plt.show()

train_size = tuple(data[num]['image'].shape[1:])
print(train_size)
