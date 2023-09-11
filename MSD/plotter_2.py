import os.path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from MSD.utils.data_utils import get_loader


colors = [
    (0, 0, 0),         # Black
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (128, 0, 0),       # Maroon
    (0, 128, 0),       # Lime
    (0, 0, 128),       # Navy
    (128, 128, 0),     # Olive
    (128, 0, 128),     # Purple
    (0, 128, 128),     # Teal
    (128, 128, 128),   # Gray
    # (255, 165, 0)      # Orange
]

models = [
    {
        "name": "DiNTS_Instance",
        "url": "64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_DiNTS_Instance_Task06_Lung",
    },
    # {
    #     "name": "DiNTS_Search",
    #     "url": "64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_DiNTS_Search_Task06_Lung",
    # },
    # {
    #     "name": "nnUNet",
    #     "url": "64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_nnUNet_Task06_Lung",
    # },
    # {
    #     "name": "SegResNetVAE",
    #     "url": "64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_SegResNetVAE_Task06_Lung",
    # },
    # {
    #     "name": "SwinUNETR48",
    #     "url": "64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_SwinUNETR48_Task06_Lung",
    # },
    # {
    #     "name": "Unetpp",
    #     "url": "64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_Unetpp_Task06_Lung",
    # },
    # {
    #     "name": "UNETR32",
    #     "url": "64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_UNETR32_Task06_Lung",
    # }
]


def colorizing(data_shape, predicted_data, target_data, original_data):
    image_pred = np.zeros((data_shape[0], data_shape[1], 3))
    image_targ = np.zeros((data_shape[0], data_shape[1], 3))
    original_slice = np.expand_dims((original_data * 255.0).astype(np.uint8), axis=2)
    original_slice = np.concatenate((original_slice, original_slice, original_slice), axis=2)

    for k in range(data_shape[0]):
        for j in range(data_shape[1]):
            image_pred[k][j] = original_slice[k][j]
            image_targ[k][j] = original_slice[k][j]
            if target_data[k][j] != 0:
                image_targ[k][j] = colors[target_data[k][j]]
            if predicted_data[k][j] != 0:
                image_pred[k][j] = colors[predicted_data[k][j]]

    image_pred = image_pred.astype(np.uint8)
    return image_pred, image_targ


def plot_data(predicted_data, original_data, target_data, base_url):
    data_shape = original_data.shape
    for i in range(data_shape[2] // 3, data_shape[2] * 7 // 8):
        data_shape = original_data.shape
        fig, axes = plt.subplots(2, 1, figsize=(5 * 1, 10))
        image_pred, image_targ = colorizing(data_shape, predicted_data[:, :, i],
                                            target_data[:, :, i], original_data[:, :, i])
        axes[0].imshow(image_pred)
        axes[0].set_title('Predicted - Slice {}'.format(i))

        image_targ = image_targ.astype(np.uint8)
        axes[1].imshow(image_targ)
        axes[1].set_title('Target - Slice {}'.format(i))

        plt.savefig(base_url + f"/{i}" + ".png")
        print(base_url + f"/{i}" + ".png")
        # plt.show()


def main_worker(gpu, args):
    loader, _, _ = get_loader(args)
    for model in models:
        base = "./res/" + model["name"] + "/"
        if not os.path.exists(base):
            os.mkdir(base)

        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            url = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            if not os.path.exists(base + url):
                os.mkdir(base + url)
            predicted_path = f'/home/amin/CETI/medical_image/SSL_VAN/runs/MSD_new/test_log/output_True_False/' \
                             f'{model["url"]}/{url}'
            predicted_img = nib.load(predicted_path)
            predicted_data = predicted_img.get_fdata().astype(np.uint8)
            plot_data(predicted_data, data[0][0].cpu().numpy(),
                      target[0][0].cpu().numpy().astype(np.uint8), base_url=base + url)
            print(base + url)


class Config:
    def __init__(self):
        self.base_data = '/media/amin/Amin/MSD-data'
        self.json_list = '../input_list/dataset_MSD_List.json'
        self.space_x = 1.0
        self.space_y = 1.0
        self.space_z = 1.0
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
        self.logdir = './runs/MSD/test_log'
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
        self.task = "Task06_Lung"
        self.num_samples = 4


def get_out_channels(args):
    if args.task == "Task01_BrainTumour":
        args.out_channels = 4
        args.in_channels = 4
    elif args.task == "Task02_Heart":  # OK
        args.out_channels = 2
    elif args.task == "Task03_Liver":  # OK !!!
        args.out_channels = 3
    elif args.task == "Task06_Lung":  # OK
        args.out_channels = 2
    elif args.task == "Task07_Pancreas":  # OK
        args.out_channels = 3
    elif args.task == "Task08_HepaticVessel":  # OK
        args.out_channels = 3
    elif args.task == "Task09_Spleen":  # OK
        args.out_channels = 2
    elif args.task == "Task10_Colon":  # OK
        args.out_channels = 2


args = Config()
get_out_channels(args)
acc = main_worker(gpu=0, args=args)

