import os
import pydicom
import nibabel as nib
import numpy as np
import json


json_url = '/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_LUNA16_List.json'
in_file = open(json_url)
json_data = json.load(in_file)
for data_type in json_data:
    for file_data in json_data[data_type]:
        url = os.path.join('/media/amin/SP PHD U3/CT_Segmentation_Images/3D/LUNA_16/manifest-1600709154662/',
                           file_data['files_dir'])
        number = int(file_data['subject_id'].split('-')[2])
        output_dir = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/LUNA_16/manifest-1600709154662/images/'
        # Get a list of all DICOM files in the input directory
        dicom_files = [os.path.join(url, f) for f in os.listdir(url) if f.endswith('.dcm')]
        dicom_files = sorted(dicom_files)
        data = []
        for f in dicom_files:
            dcm = pydicom.dcmread(f)
            data.append(dcm.pixel_array)

        image = nib.Nifti1Image(np.array(data), np.eye(4))

        # Save the NIfTI image to a file
        output_file = os.path.join(output_dir, 'img_{}.nii.gz'.format(number))
        nib.save(image, output_file)
        print(output_file)

# # Load the NIfTI image from file
# loaded_image = nib.load(output_file)
#
# # Get the image data as a numpy array
# image_data = loaded_image.get_fdata()

