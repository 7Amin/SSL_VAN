import torch
from pretrain.config.masking import PATCHES
import numpy as np
import random


def create_mask(phi_1, phi_2, mask_length, tensor_size):
    # tensor_size is z, x, y
    selected_images = np.random.choice(tensor_size[0], int(phi_1 * tensor_size[0]), replace=False)

    mask = torch.ones(tensor_size[0], tensor_size[1], tensor_size[2])
    masked_images_index = []
    for selected_image in selected_images:
        print(f"selected_image: {selected_image}")
        for image_idx in range(max(0, selected_image - mask_length + 1), selected_image + 1):
            # Select a patch size randomly
            patch_size = np.random.choice(PATCHES)
            print(f"image_idx: {image_idx}, patch_size: {patch_size}")
            if (tensor_size[1] < patch_size or tensor_size[2] < patch_size) and not (image_idx in masked_images_index):
                mask[image_idx] = 0
            elif not (image_idx in masked_images_index):
                num_patches_x = tensor_size[1] // patch_size
                num_patches_y = tensor_size[2] // patch_size

                for patch_idx_x in range(num_patches_x):
                    for patch_idx_y in range(num_patches_y):
                        if random.random() < phi_2:
                            patch_start_x = patch_idx_x * patch_size
                            patch_start_y = patch_idx_y * patch_size
                            patch_end_x = (patch_idx_x + 1) * patch_size
                            patch_end_y = (patch_idx_y + 1) * patch_size
                            mask[image_idx, patch_start_x:patch_end_x, patch_start_y:patch_end_y] = 0
            else:
                print(image_idx)
            masked_images_index.append(image_idx)
    return mask.unsqueeze(0)


def apply_mask(data, args):
    masks = []
    batch, c, z, x, y = data.shape

    for _ in range(batch):
        mask_channels = []
        for t in range(c):
            mask = create_mask(args.phi_1, args.phi_2, args.mask_length, (z, x, y))
            mask_channels.append(mask)
        merged_tensor = torch.cat(mask_channels, dim=0)
        masks.append(merged_tensor)
    masks_tensor = torch.stack(masks).to('cuda')
    return data * masks_tensor


# Test the create_mask function
# phi_1 = 0.3
# phi_2 = 0.7
# mask_length = 5
# tensor_size = (20, 256, 64)
#
# mask = create_mask(phi_1, phi_2, mask_length, tensor_size)
# print(mask.shape)
#
# data = torch.randn(6, 4, 20, 64, 32).to('cuda')
# class ARGS:
#     def __init__(self):
#         self.phi_1 = 0.7
#         self.phi_2 = 0.3
#         self.mask_length = 4
# apply_mask(data, ARGS())

