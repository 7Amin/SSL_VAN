import os
import pydicom
import pandas as pd
import nibabel as nib
import numpy as np
import json


def part_one():
    csv_url = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/TCIA_LIDC/manifest-1600709154662/metadata.csv'
    df = pd.read_csv(csv_url)

    json_url = '/home/amin/CETI/medical_image/SSL_VAN/jsons/dataset_LIDC_0.json'
    in_file = open(json_url)
    json_data = json.load(in_file)
    result = dict()
    for data_type in json_data:
        res = []
        for file_data in json_data[data_type]:
            number = file_data['image'].split('/')[1].split('-')[2][:4]
            Subject_ID = f'LIDC-IDRI-{number}'
            df_temp = df[df['Subject ID'] == Subject_ID].sort_values(by='Number of Images', ascending=False)
            res.append({
                "name": file_data['image'],
                "files_dir": df_temp.iloc[0]['File Location'],
                "class_name": df_temp.iloc[0]['SOP Class Name'],
                "number_image": str(df_temp.iloc[0]['Number of Images']),
                "subject_id": Subject_ID,
                "study_uid": df_temp.iloc[0]['Study UID'],
                "series_uid": df_temp.iloc[0]['Series UID'],
            })

        result[data_type] = res

    with open('/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_LIDC_List.json', "w") as outfile:
        json.dump(result, outfile, indent=4)


def part_two():
    json_url = '/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_LIDC_List.json'
    in_file = open(json_url)
    json_data = json.load(in_file)
    for data_type in json_data:
        for file_data in json_data[data_type]:
            url = os.path.join('/media/amin/SP PHD U3/CT_Segmentation_Images/3D/TCIA_LIDC/manifest-1600709154662',
                               file_data['files_dir'])
            number = int(file_data['subject_id'].split('-')[2])
            output_dir = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/TCIA_LIDC/manifest-1600709154662/images/'
            output_file = os.path.join(output_dir, 'img_{}.nii.gz'.format(number))
            if os.path.isfile(output_file):
                print(number)
                continue

            # Get a list of all DICOM files in the input directory
            dicom_files = [os.path.join(url, f) for f in os.listdir(url) if f.endswith('.dcm')]
            dicom_files = sorted(dicom_files)
            data = []
            for f in dicom_files:
                dcm = pydicom.dcmread(f)
                data.append(dcm.pixel_array)

            image = nib.Nifti1Image(np.array(data), np.eye(4))

            # Save the NIfTI image to a file

            nib.save(image, output_file)
            print(output_file)


part_two()
