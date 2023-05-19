import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


color_map = [(0, 0, 0),     # Background (black)
             (128, 0, 0),   # Category 1 (red)
             (0, 128, 0),   # Category 2 (green)
             (0, 0, 128),   # Category 3 (blue)
             (128, 128, 0), # Category 4 (olive)
             (0, 128, 128), # Category 5 (teal)
             (128, 0, 128), # Category 6 (purple)
             (128, 128, 128), # Category 7 (gray)
             (64, 0, 0),    # Category 8 (dark red)
             (0, 64, 0),    # Category 9 (dark green)
             (0, 0, 64),    # Category 10 (dark blue)
             (64, 64, 0),   # Category 11 (dark olive)
             (0, 64, 64),   # Category 12 (dark teal)
             (64, 0, 64)]   # Category 13 (dark purple)


def plot_ans_save_segmentation(pred, target, input, ind):
    c, seq, w, h = pred[0].shape
    b = len(pred)
    for i in range(b):
        print(i)
        img = pred[i].cpu().numpy()
        targ = target[i].cpu().numpy()
        inp = input[i][0].cpu().numpy()
        sample_stack(inp, f"normal_data_{ind}_{i}")
        img_map = np.argmax(img, axis=0).astype(np.int16)
        targ_map = np.argmax(targ, axis=0).astype(np.int16)

        new_img_map = convert_colored(img_map)
        sample_stack(new_img_map, f"predicted image {ind}_{i}")
        new_targ_map = convert_colored(targ_map)
        sample_stack(new_targ_map, f"target image {ind}_{i}")

        # sitk_img = sitk.GetImageFromArray(img_map)
        # sitk_img_overlay = sitk.LabelMapContourOverlay(sitk_img, sitk_img,
        #                                                colormap=color_map,
        #                                                opacity=1)
        # image_filename = f"image_pred_{idx}_{i}.png"
        # sitk.WriteImage(sitk_img_overlay, image_filename)
        #
        # sitk_targ = sitk.GetImageFromArray(targ_map)
        # sitk_targ_overlay = sitk.LabelMapContourOverlay(sitk_targ, sitk_targ,
        #                                                 colormap=color_map,
        #                                                 opacity=1,
        #                                                 backgroundValue=0)
        # targ_filename = f"image_target_{idx}_{i}.png"
        # sitk.WriteImage(sitk_targ_overlay, targ_filename)


def convert_colored(images):
    seq, w, h = images.shape
    new_image = np.zeros((seq, w, h, 3))
    for i in range(seq):
        for j in range(w):
            for k in range(h):
                new_image[i, j, k] = color_map[images[i, j, k]]
    return new_image


def sample_stack(stack, title="normal image", rows=12, cols=8):
    fig, ax = plt.subplots(rows, cols, figsize=[18, 20])
    for i in range(rows * cols):
        ax[int(i/cols), int(i % cols)].set_title(f'slice {i}')
        ax[int(i/cols), int(i % cols)].imshow(stack[i], cmap='gray')
        ax[int(i/cols), int(i % cols)].axis('off')
    plt.title(title)
    plt.savefig(f'./res/{title}.png')
    plt.show()
