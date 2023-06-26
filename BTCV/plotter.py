import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

image_name = '0039.nii.gz'

# Load the original image
original_path = f'/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training/img/img{image_name}'
original_img = nib.load(original_path)
original_data = original_img.get_fdata()
original_data = (original_data - original_data.min()) / (original_data.max() - original_data.min())
original_shape = original_data.shape

# Load the predicted data
predicted_path = f'/home/amin/CETI/medical_image/SSL_VAN/runs/BTCV_new/test_log/output_True_False' \
                 f'/64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_VANV6GL_2/img{image_name}'
predicted_img = nib.load(predicted_path)
predicted_data = predicted_img.get_fdata()

# Load the target data
target_path = f'/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training/label/label{image_name}'
target_img = nib.load(target_path)
target_data = target_img.get_fdata()

output_shape = original_data.shape
predicted_data = ndi.zoom(predicted_data, (output_shape / np.array(predicted_data.shape).astype(float)).astype(float), order=1)


cmap = plt.cm.get_cmap('tab20', 14)  # 14 is the number of unique voxel values (0 to 13)

cmap.set_under('black')

slice_indices = []
base = 50
for i in range(20):
    slice_indices.append(i * 2 + base)
num_slices = len(slice_indices)

fig, axes = plt.subplots(2, num_slices + 1, figsize=(5*num_slices, 10))

for i, idx in enumerate(slice_indices):
    # Get the predicted and target slices
    predicted_slice = np.rot90(predicted_data[:, :, idx])
    target_slice = np.rot90(target_data[:, :, idx], k=1)
    target_slice = np.flip(target_slice, axis=-1)
    original_data[:, :, idx] = np.rot90(original_data[:, :, idx], k=1)
    original_data[:, :, idx] = np.flip(original_data[:, :, idx], axis=-1)

    # Plot the predicted slice
    axes[0, i].imshow(predicted_slice, cmap=cmap, vmin=1, vmax=13)  # Exclude background value 0
    axes[0, i].set_title('Predicted - Slice {}'.format(idx))

    # Plot the target slice
    axes[1, i].imshow(target_slice, cmap=cmap, vmin=1, vmax=13)  # Exclude background value 0
    axes[1, i].set_title('Target - Slice {}'.format(idx))


for i in range(num_slices):
    axes[0, i + 1].imshow(original_data[:, :, slice_indices[i]], cmap='gray', alpha=0.6)
    axes[1, i + 1].imshow(original_data[:, :, slice_indices[i]], cmap='gray', alpha=0.6)

cbar = fig.colorbar(axes[0, 0].imshow(np.zeros((10, 10)), cmap=cmap, vmin=1, vmax=13))
# Set the colorbar label
cbar.set_label('Colorbar Label')

# Create a legend
legend_labels = ['Original Data']
legend_handles = [plt.Line2D([0], [0], color='gray', alpha=0.6, lw=4)]
axes[0, 0].legend(legend_handles, legend_labels)

plt.savefig(image_name.replace('.', '_') + ".png")
plt.show()

