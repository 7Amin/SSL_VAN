import torch
from pretrain.config.masking import PATCHES
import numpy as np


def create_mask(phi_1, phi_2, mask_length, tensor_size):
    # tensor_size is z, x, y
    # Randomly select images with probability phi_1
    selected_images = np.random.choice(tensor_size[0], int(phi_1 * tensor_size[0]), replace=False)

    mask = torch.ones(tensor_size[0], tensor_size[1], tensor_size[2])
    masked_images_index = []
    for selected_image in selected_images:
        print(f"selected_image: {selected_image}")
        for image_idx in range(max(0, selected_image - mask_length + 1), selected_image + 1):
            # Select a patch size randomly
            patch_size = np.random.choice(PATCHES)
            print(f"image_idx: {image_idx}, patch_size: {patch_size}")
            if tensor_size[1] < patch_size or tensor_size[2] < patch_size:
                # If the tensor size is smaller than the patch size, set the entire image as masked
                mask[image_idx] = 0
            elif not (image_idx in masked_images_index):

            # Calculate the number of patches for the current image
                num_patches_x = tensor_size[1] // patch_size
                num_patches_y = tensor_size[2] // patch_size

                # Randomly select patches with probability phi_2
                selected_patches_x = np.random.choice(num_patches_x, int(phi_2 * num_patches_x), replace=False)
                selected_patches_y = np.random.choice(num_patches_y, int(phi_2 * num_patches_y), replace=False)

                # Mask the selected patches
                for patch_idx_x in selected_patches_x:
                    for patch_idx_y in selected_patches_y:
                        patch_start_x = patch_idx_x * patch_size
                        patch_start_y = patch_idx_y * patch_size
                        patch_end_x = (patch_idx_x + 1) * patch_size
                        patch_end_y = (patch_idx_y + 1) * patch_size
                        mask[image_idx, patch_start_x:patch_end_x, patch_start_y:patch_end_y] = 0
            else:
                print(image_idx)
            masked_images_index.append(image_idx)

    return mask


# Test the create_mask function
# phi_1 = 0.3
# phi_2 = 0.7
# mask_length = 5
# tensor_size = (20, 256, 64)
#
# mask = create_mask(phi_1, phi_2, mask_length, tensor_size)
print(mask.shape)
