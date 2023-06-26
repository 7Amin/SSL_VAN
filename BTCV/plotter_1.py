import os
import time
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


from BTCV.utils.data_utils import get_loader


def plot_data(predicted_data, original_data, target_data, image_name):
    cmap = plt.cm.get_cmap('tab20', 14)  # 14 is the number of unique voxel values (0 to 13)
    # cmap.set_under('black')

    slice_indices = []
    base = 130
    for i in range(20):
        slice_indices.append(i * 4 + base)
    num_slices = len(slice_indices)

    fig, axes = plt.subplots(2, num_slices, figsize=(5 * num_slices, 10))

    for i, idx in enumerate(slice_indices):
        # Get the predicted and target slices
        # predicted_slice = np.rot90(predicted_data[:, :, idx])
        # target_slice = np.rot90(target_data[:, :, idx])
        # original_data[:, :, idx] = np.rot90(original_data[:, :, idx])

        axes[0, i].imshow(original_data[:, :, slice_indices[i]], cmap='gray', alpha=0.6)
        axes[1, i].imshow(original_data[:, :, slice_indices[i]], cmap='gray', alpha=0.6)
        predicted_slice = predicted_data[:, :, idx]
        target_slice = target_data[:, :, idx]
        original_data[:, :, idx] = original_data[:, :, idx]

        # Plot the predicted slice
        axes[0, i].imshow(predicted_slice, cmap=cmap, vmin=1, vmax=13)  # Exclude background value 0
        axes[0, i].set_title('Predicted - Slice {}'.format(idx))

        # Plot the target slice
        axes[1, i].imshow(target_slice, cmap=cmap, vmin=1, vmax=13)  # Exclude background value 0
        axes[1, i].set_title('Target - Slice {}'.format(idx))


    plt.savefig(image_name.replace('.', '_') + ".png")
    plt.show()

def main_worker(gpu, args):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    loader = get_loader(args)
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        url = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
        predicted_path = f'/home/amin/CETI/medical_image/SSL_VAN/runs/BTCV_new/test_log/output_True_False' \
                         f'/64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_VANV6GL_2/{url}'
        predicted_img = nib.load(predicted_path)
        predicted_data = predicted_img.get_fdata().astype(np.uint8)
        plot_data(predicted_data, data[0][0].cpu().numpy(),
                  target[0][0].cpu().numpy().astype(np.uint8), url)
        print(url)


class Config:
    def __init__(self):
        self.base_data = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training'
        self.json_list = '../input_list/dataset_BTCV_List.json'
        self.space_x = 1.5
        self.space_y = 1.5
        self.space_z = 2.0
        self.a_min = -175.0
        self.a_max = 250.0
        self.b_min = 0.0
        self.b_max = 1.0
        self.roi_x = 96
        self.roi_y = 96
        self.roi_z = 96
        self.batch_size = 1
        self.workers = 1
        self.RandFlipd_prob = 0.2
        self.RandRotate90d_prob = 0.2
        self.RandScaleIntensityd_prob = 0.1
        self.RandShiftIntensityd_prob = 0.1
        self.test_mode = True
        self.distributed = False
        self.logdir = './runs/BTCV/test_log'
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
        self.valid_loader = ""


acc = main_worker(gpu=0, args=Config())

