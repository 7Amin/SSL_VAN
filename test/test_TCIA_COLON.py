import os
import pydicom
import pandas as pd
import nibabel as nib
import numpy as np
import json
import random
import warnings
import glob


def part_one():
    csv_url = '/media/amin/Amin/CT_Segmentation_Images/3D/TCIA_CT_Colonography_Trial/metadata.csv'
    df = pd.read_csv(csv_url)

    json_url = '/home/amin/CETI/medical_image/SSL_VAN/jsons/dataset_TCIAcolon_v2_0.json'
    in_file = open(json_url)
    json_data = json.load(in_file)
    result = dict()
    for data_type in json_data:
        res = []
        for file_data in json_data[data_type]:
            number = int(file_data['image'].split('/')[1].split('.')[0][4:]) - 2
            # while len(number) < 4:
            #     number = "0" + number
            if len(df) <= number or number < 0:
                print(number)
                continue
            df_temp = df.iloc[number]

            res.append({
                "name": file_data['image'],
                "files_dir": df_temp['File Location'],
                "class_name": df_temp['SOP Class Name'],
                "number_image": str(df_temp['Number of Images']),
                "subject_id": df_temp['Subject ID'],
                "study_uid": df_temp['Study UID'],
                "series_uid": df_temp['Series UID'],
            })

        result[data_type] = res

    with open('/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_TCIAcolon_List.json', "w") as outfile:
        json.dump(result, outfile, indent=4)


def part_one_new():
    csv_url = '/media/amin/Amin/CT_Segmentation_Images/3D/TCIA_CT_Colonography_Trial/metadata.csv'
    df = pd.read_csv(csv_url)
    df = df[df['Number of Images'] > 5]
    df = df[df['SOP Class Name'] == 'CT Image Storage']
    result = dict()
    res = []
    for index, row in df.iterrows():
        if row['Number of Images'] < 96:
            continue
        res.append({
            "name": 'images/img_{}.nii.gz'.format(index + 1),
            "files_dir": row['File Location'],
            "class_name": row['SOP Class Name'],
            "number_image": str(row['Number of Images']),
            "subject_id": row['Subject ID'],
            "study_uid": row['Study UID'],
            "series_uid": row['Series UID'],
        })

    random.shuffle(res)
    result['training'] = res[:1550]
    result['validation'] = res[1550:]

    with open('/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_TCIAcolon_List.json', "w") as outfile:
        json.dump(result, outfile, indent=4)


def part_two():
    json_url = '/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_TCIAcolon_List.json'
    in_file = open(json_url)
    json_data = json.load(in_file)
    count = 0
    for data_type in json_data:
        for file_data in json_data[data_type]:
            file_dir = file_data['files_dir'].replace('\\', '/')
            url = os.path.join('/media/amin/Amin/CT_Segmentation_Images/3D/TCIA_CT_Colonography_Trial', file_dir)
            counter = len(glob.glob1(url, "*.dcm"))
            if not (os.path.exists(url) and counter == int(file_data['number_image'])):
                continue
            output_dir = '/media/amin/Amin/CT_Segmentation_Images/3D/TCIA_CT_Colonography_Trial/'
            output_file = os.path.join(output_dir, file_data['name'])
            count += 1
            print(f"count is {count}")
            if os.path.isfile(output_file):
                print(file_data['name'])
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
    json_url = '/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_TCIAcolon_List.json'
    in_file = open(json_url)
    json_data1 = json.load(in_file)

    result = dict()
    for data_type in json_data1:
        res = []
        for file_data in json_data1[data_type]:
            res.append({
                "image": file_data['name']}
            )
        result[data_type] = res
    with open('/home/amin/CETI/medical_image/SSL_VAN/jsons/dataset_TCIAcolon_v2_0.json', "w") as outfile:
        json.dump(result, outfile, indent=4)


def part_four():
    json_url = './jsons/dataset_TCIAcolon_v2_0.json'
    # json_url = '/home/amin/CETI/medical_image/SSL_VAN/jsons/dataset_TCIAcolon_v2_0.json'
    in_file = open(json_url)
    base_url = "/home/karimimonsefi.1/images/Colonography/"
    # base_url = "/media/amin/Amin/CT_Segmentation_Images/3D/TCIA_CT_Colonography_Trial/images/"
    json_data = json.load(in_file)
    for data_type in json_data:
        for file_data in json_data[data_type]:
            url = base_url + file_data["image"]
            warnings.warn(f"Start: url is {url}")
            try:
                loaded_image = nib.load(url)
                image_data = loaded_image.get_fdata()
                warnings.warn(f"Done: {url}")
            except:
                warnings.warn(f"Failed: has problem with {url}")


part_four()
