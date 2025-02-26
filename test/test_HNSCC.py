import os
import pydicom
import pandas as pd
import nibabel as nib
import numpy as np
import json


def part_one():
    csv_url = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/HNSCC/manifest-1600709154662/metadata.csv'
    df = pd.read_csv(csv_url)

    json_url = '/home/amin/CETI/medical_image/SSL_VAN/jsons/dataset_HNSCC_0.json'
    in_file = open(json_url)
    json_data = json.load(in_file)
    result = dict()
    for data_type in json_data:
        res = []
        for file_data in json_data[data_type]:
            number = file_data['image'].split('/')[1].split('.')[0][4:]
            while len(number) < 4:
                number = "0" + number
            Subject_ID = f'LIDC-IDRI-{number}'
            df_temp = df[df['Subject ID'] == Subject_ID].sort_values(by='Number of Images', ascending=False)
            if len(df_temp) < 1:
                continue
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

    with open('/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_HNSCC_List.json', "w") as outfile:
        json.dump(result, outfile, indent=4)


def part_two():
    json_url = '/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_HNSCC_List.json'
    in_file = open(json_url)
    json_data = json.load(in_file)
    count = 0
    for data_type in json_data:
        for file_data in json_data[data_type]:
            url = os.path.join('/media/amin/SP PHD U3/CT_Segmentation_Images/3D/HNSCC/manifest-1600709154662',
                               file_data['files_dir'])
            number = int(file_data['subject_id'].split('-')[2])
            # output_dir = '/media/amin/SP PHD U3/CT_Segmentation_Images/3D/HNSCC/manifest-1600709154662/images/'
            output_dir = '/media/amin/ADATA_Amin/CT_Segmentation_Images/3D/HNSCC/manifest-1600709154662/images/'
            # output_file = os.path.join(output_dir, 'img_{}.nii.gz'.format(number))
            output_file = os.path.join(output_dir, 'img_{}.nii.gz'.format(number))
            count += 1
            print(f"count is {count}")
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


def part_three():
    json_url = '/home/amin/CETI/medical_image/SSL_VAN/jsons/dataset_HNSCC_0.json'
    in_file = open(json_url)
    json_data = json.load(in_file)
    result = dict()
    for data_type in json_data:
        res = []
        for file_data in json_data[data_type]:
            number = int(file_data['image'].split('_')[1].split('.')[0])
            if number > 1012:
                continue
            res.append(file_data)
        result[data_type] = res
    with open(json_url, "w") as outfile:
        json.dump(result, outfile, indent=4)


def part_four():
    json_url = '/home/amin/CETI/medical_image/SSL_VAN/jsons/dataset_HNSCC_0.json'
    in_file = open(json_url)
    base_url = "/media/amin/Amin/CT_Segmentation_Images/3D/HNSCC/"
    json_data = json.load(in_file)
    for data_type in json_data:
        for file_data in json_data[data_type]:
            url = base_url + file_data["image"]
            try:
                loaded_image = nib.load(url)
                image_data = loaded_image.get_fdata()
                print("Done")
            except:
                print(url)


part_four()
