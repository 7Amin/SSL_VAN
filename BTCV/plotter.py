import numpy as np
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt

image_name = '0036.nii.gz'

# Load the original image
original_path = f'/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training/img/img{image_name}'
original_img = nib.load(original_path)
original_data = original_img.get_fdata()

# Load the predicted data
predicted_path = f'/home/amin/CETI/medical_image/SSL_VAN/runs/BTCV_new/test_log/output_True_False' \
                 f'/64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_VANV5GL_2/img{image_name}'

predicted_img = nib.load(predicted_path)
predicted_data = predicted_img.get_fdata()

# Load the target data
target_path = f'/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BTCV/Abdomen/RawData/Training/label/label{image_name}'
target_img = nib.load(target_path)
target_data = target_img.get_fdata()


# Define the desired spacing
spacing = (1.5, 1.5, 2.0)  # Replace with the desired spacing values

# Create the Spacingd transform
spacing_transform = tio.Resample(spacing)

# Apply the transform to the original and target images
original_resampled = spacing_transform(tio.ScalarImage(tensor=original_data))
target_resampled = spacing_transform(tio.LabelMap(tensor=target_data))

# Get the resampled data
original_data_resampled = original_resampled.numpy()
target_data_resampled = target_resampled.numpy()

# Define the colormap for voxel values
cmap = plt.cm.get_cmap('tab20', 14)  # 14 is the number of unique voxel values (0 to 13)

# Set the background (0) value to black
cmap.set_under('black')

# Plot the desired slices for comparison
slice_indices = [125, 150, 175]  # Example slice indices to plot
num_slices = len(slice_indices)

fig, axes = plt.subplots(1, num_slices, figsize=(5*num_slices, 5))

for i, idx in enumerate(slice_indices):
    # Get the original, predicted, and target slices
    original_slice = np.rot90(original_data[:, :, idx])
    predicted_slice = np.rot90(predicted_data[:, :, idx])
    target_slice = np.rot90(target_data[:, :, idx])

    # Apply the colormap to the predicted and target slices
    predicted_colored = cmap(predicted_slice)
    target_colored = cmap(target_slice)

    # Set the background color to black
    predicted_colored[predicted_slice == 0] = [0, 0, 0, 1]
    target_colored[target_slice == 0] = [0, 0, 0, 1]

    # Combine the original, predicted, and target slices
    combined_slice = original_slice + predicted_colored + target_colored

    axes[i].imshow(combined_slice)
    axes[i].set_title('Combined - Slice {}'.format(idx))

plt.show()

