import os
import shutil
import tempfile
import cv2
import numpy as np

import matplotlib.pyplot as plt
from BRATS21.utils.data_utils import get_loader

class Config:
    def __init__(self):
        self.base_data = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BraTS21'
        self.json_list = '../input_list/dataset_BRATS21_List.json'
        self.space_x = 1.0
        self.space_y = 1.0
        self.space_z = 1.0
        self.a_min = -100.0
        self.a_max = 2000.0
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


def convert_rgb(image):
    width, height, c = image.shape
    for y in range(height):
        for x in range(width):
            r, g, b = image[x, y]
            if (r, g, b) == (255, 255, 255):
                image[x, y] = (0, 0, 255)
            elif (r, g, b) == (255, 255, 0):
                image[x, y] = (255, 0, 0)
            elif (r, g, b) == (255, 0, 255):
                image[x, y] = (0, 0, 255)
            elif (r, g, b) == (0, 255, 255):
                image[x, y] = (0, 255, 0)
    return image


args = Config()
_, train_ds, val_ds = get_loader(args)
data = train_ds[145]
# data = val_ds[120]
# data = val_ds
slice_id = 62
t = 10
num = 0
# num = 20
scale_sum = 0.0
scale_divide = 1.0


# pick one image from DecathlonDataset to visualize and check the 4 channels
print(f"image shape: {data[num]['image'].shape}")
for o in range(t):
    plt.figure("image", (24, 6))
    for i in range(4):
        # a = (((data[num]["image"][i, :, :, slice_id + (o + 1) * 5].detach().cpu() + 0.0) / 1.0).numpy() * 255.0).astype(np.int32)
        # a = a[50: 135 + 50, 30: 145 + 70]
        # cv2.imwrite(f'input_slide_{o + 1}_channel_{i}.png', a)
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i} for slice {slice_id + (o + 1) * 5}")
        plt.imshow((data[num]["image"][i, :, :, slice_id + (o + 1) * 5].detach().cpu() + scale_sum) / scale_divide,  cmap="gray")
        d = ((data[num]["image"][i, :, :, slice_id + (o + 1) * 5].detach().cpu() + scale_sum) / scale_divide).numpy()
        print(f"for i = {i} => min is {d.min()} and max is {d.max()}")

    # plt.savefig(f'slide_{o + 1}.png')
    plt.show()
    print("-----------------------------")

print(f"label shape: {data[num]['label'].shape}")

for o in range(t):
    plt.figure("label", (24, 6))
    img = []
    for i in range(3):
        # a = (data[num]["label"][i, :, :, slice_id + (o + 1) * 5].detach().cpu().numpy() * 255.0).astype(np.int32)
        # a = a[50: 135 + 50, 30: 145 + 70]
        # img.append(np.expand_dims(a, axis=2))

        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i} for slice {slice_id + (o + 1) * 5}")
        plt.imshow(data[num]["label"][i, :, :, slice_id + (o + 1) * 5].detach().cpu())
        d = data[num]["label"][i, :, :, slice_id + (o + 1) * 5].detach().cpu().numpy()
        print(f"for i = {i} => min is {d.min()} and max is {d.max()}")


    # lab = np.concatenate((img[0], img[1], img[2]), axis=2)
    # lab = convert_rgb(lab)
    # cv2.imwrite(f'label_slide_{o + 1}.png', lab)
    # plt.savefig(f'label_3_slide_{o + 1}.png')
    plt.show()
    print("-----------------------------")


train_size = tuple(data[num]['image'].shape[1:])
print(train_size)
