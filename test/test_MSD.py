import os
import pydicom
import nibabel as nib
import numpy as np
import json
import random


tasks_list = ["Task01_BrainTumour", "Task02_Heart", "Task03_Liver", "Task04_Hippocampus", "Task05_Prostate",
              "Task06_Lung", "Task07_Pancreas", "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"]
base_url = "/media/amin/Amin/MSD-data"


def split_list(lst, rate=0.8):
    n = len(lst)
    split_index = int(rate * n)  # Index to split the list
    random.shuffle(lst)  # Shuffle the list randomly

    # Split the list into two parts
    first_part = lst[:split_index]
    second_part = lst[split_index:]

    return first_part, second_part


def first():
    res = dict()
    for task_name in tasks_list:
        json_path = os.path.join(base_url, task_name, "dataset.json")
        in_file = open(json_path)
        json_data = json.load(in_file)
        train_data = json_data['training']
        test = json_data['test']
        training, validation = split_list(train_data)
        res[task_name] = {
            "training": training,
            "validation": validation,
            "test": test
        }

    with open('/home/amin/CETI/medical_image/SSL_VAN/input_list/dataset_MSD.json', "w") as outfile:
        json.dump(res, outfile, indent=4)


first()
