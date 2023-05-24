import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from BRATS21.utils.data_utils import get_loader


loader = get_loader(args)
val_loader = loader[1]
for idx, batch_data in enumerate(val_loader):
    if isinstance(batch_data, list):
        data, target = batch_data
    else:
        data, target = batch_data["image"], batch_data["label"]
    slice_id = 32
    # pick one image from DecathlonDataset to visualize and check the 4 channels
    print(f"image shape: {val_ds[2]['image'].shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_ds[2]["image"][i, :, :, slice_id].detach().cpu(),  cmap="gray") #
    plt.show()
    # also visualize the 3 channels label corresponding to this image
    print(f"label shape: {val_ds[2]['label'].shape}")
    plt.figure("label", (24, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, slice_id].detach().cpu())
    plt.show()

    train_size = tuple(val_ds[6]['image'].shape[1:])
    print(train_size)
