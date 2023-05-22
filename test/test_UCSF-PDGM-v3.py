import os
import pydicom
import pandas as pd
import nibabel as nib
import numpy as np
import json
import random
import warnings
import glob
import shutil


def part_one():
    csv_url = '/media/amin/Amin/CT_Segmentation_Images/3D/Brain/UCSF-PDGM-v3/UCSF-PDGM-metadata_v2.csv'
    df = pd.read_csv(csv_url)
    df = df[df['BraTS21 ID'] != df['BraTS21 ID']]
    result = dict()
    res = []
    # base = "/media/amin/Amin/CT_Segmentation_Images/3D/Brain/UCSF-PDGM-v3/UCSF-PDGM-v3/"
    for index, row in df.iterrows():
        r = row['ID'][:10] + '0' + row['ID'][10:]
        u = r + "_nifti/"
        res.append({
            "image": [
                u + f"{r}_FLAIR.nii.gz",
                u + f"{r}_T1.nii.gz",
                u + f"{r}_T1c.nii.gz",
                u + f"{r}_T2.nii.gz",
            ],
            "label": u + f"{r}_tumor_segmentation.nii.gz",
        })

    result["training"] = res

    with open('/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_UCSF_PDGM_List.json', "w") as outfile:
        json.dump(result, outfile, indent=4)


def part_two():
    csv_url = '/media/amin/Amin/CT_Segmentation_Images/3D/Brain/UCSF-PDGM-v3/UCSF-PDGM-metadata_v2.csv'
    df = pd.read_csv(csv_url)
    df = df[df['BraTS21 ID'] != df['BraTS21 ID']]
    base = "/media/amin/Amin/CT_Segmentation_Images/3D/Brain/UCSF-PDGM-v3/UCSF-PDGM-v3/"
    for index, row in df.iterrows():
        r = row['ID'][:10] + '0' + row['ID'][10:]
        if "0289" in r:
            continue
        print(r)
        u = r + "_nifti/"

        temp = [u + f"{r}_FLAIR.nii.gz",
                u + f"{r}_T1.nii.gz",
                u + f"{r}_T1c.nii.gz",
                u + f"{r}_T2.nii.gz",
                u + f"{r}_tumor_segmentation.nii.gz"]

        for t in temp:
            url = "/media/amin/Amin/CT_Segmentation_Images/3D/Brain/UCSF-PDGM-v3/filtered_data/" + u
            if not os.path.exists(url):
                os.mkdir(url)
            src_path = base + t
            dst_path = "/media/amin/Amin/CT_Segmentation_Images/3D/Brain/UCSF-PDGM-v3/filtered_data/" + t
            # shutil.move(, )
            shutil.copyfile(src_path, dst_path)


def part_three():
    json_url = '/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_UCSF_PDGM_List.json'
    in_file = open(json_url)
    base_url = "/media/amin/Amin/CT_Segmentation_Images/3D/Brain/UCSF-PDGM-v3/filtered_data/"
    json_data = json.load(in_file)
    for data_type in json_data:
        for file_data in json_data[data_type]:
            for i in range(4):
                url = base_url + file_data["image"][i]
                warnings.warn(f"Start: url is {url}")
                try:
                    url = "/media/amin/SP PHD U3/CT_Segmentation_Images/3D/BraTS21/TrainingData/BraTS2021_00133/BraTS2021_00133_t1.nii.gz"
                    loaded_image = nib.load(url)
                    image_data = loaded_image.get_fdata()
                    warnings.warn(f"Done: {url}")
                except:
                    warnings.warn(f"Failed: has problem with {url}")
            url = base_url + file_data["label"]
            warnings.warn(f"Start: url is {url}")
            try:
                loaded_image = nib.load(url)
                image_data = loaded_image.get_fdata()
                warnings.warn(f"Done: {url}")
            except:
                warnings.warn(f"Failed: has problem with {url}")


part_three()
