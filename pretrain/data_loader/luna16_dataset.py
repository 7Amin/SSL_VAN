import json
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from pydicom import dcmread

from data_cleaner.segmentation_luna import get_mask_of_subject


def _read_images_directories(args, data_type):
    base_url = args.base_dir_code
    json_url = base_url + args.luna16_json
    in_file = open(json_url)
    json_data = json.load(in_file)
    return json_data[data_type]


def _load_images(args, subject_info):
    x = args.size_x
    y = args.size_y
    luda16_data = os.path.join(args.base_data + args.luna_data, subject_info['files_dir'])
    paths = os.listdir(luda16_data)
    result_images = []
    result_labels = []
    xml_url = ""
    images = dict()
    for path in paths:
        temp_url = os.path.join(luda16_data, path)
        if os.path.isfile(temp_url) and temp_url.endswith('.dcm'):
            ds = dcmread(temp_url)
            key = float(ds.SliceLocation)
            image = (ds.pixel_array + 5000) / 10000.0
            images[key] = image
        elif os.path.isfile(temp_url) and temp_url.endswith('.xml'):
            xml_url = temp_url

    my_keys = list(images.keys())
    my_keys.sort()
    masked_images = get_mask_of_subject(xml_url)
    for key in my_keys:
        temp = cv2.resize(images[key], (x, y), interpolation=cv2.INTER_AREA)
        result_images.append(temp)
        if key in masked_images:
            temp = cv2.resize(masked_images[key], (x, y), interpolation=cv2.INTER_AREA)
            result_labels.append(temp)
        else:
            result_labels.append(np.zeros((x, y)))

    return np.array(result_images), np.array(result_labels)


class Luna16Dataset(Dataset):
    def __init__(self, args, data_type="training"):
        self.args = args
        self.subjects_info = _read_images_directories(args, data_type)

    def __len__(self):
        return len(self.subjects_info)

    def __getitem__(self, index):
        subject_info = self.subjects_info[index]
        data, label = _load_images(self.args, subject_info)
        return torch.from_numpy(data), torch.from_numpy(label)
