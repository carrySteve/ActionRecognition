# -*- coding: utf-8 -*-

import json
import sys
import math

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from io_common import FileProgressingbar, generate_lineidx, img_from_base64
from tsv_io import TSVFile

TRAIN_PATH = "../data/train.tsv"
VAL_PATH = "../data/val.tsv"

# TRAIN_PATH = "../data_tsv/train.tsv"
# VAL_PATH = "../data_tsv/val.tsv"
# TEST_PATH = "../data/test.tsv"

trans = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize([0.6378, 0.4613, 0.4617], [0.2851, 0.1988, 0.3424])
])

NUM_FRAMES = 10


class VolleyballDataset(Dataset):
    def __init__(self, phase):
        if phase == "train":
            self.tsv_path = TRAIN_PATH
        elif phase == 'val':
            self.tsv_path = VAL_PATH
        else:
            self.tsv_path = TEST_PATH

        self.tsv = TSVFile(self.tsv_path)

    def __getitem__(self, idx):
        row = self.tsv.seek(idx)
        set_id = row[0]
        tar_frame = row[1]
        gact = row[2]
        group_info = (set_id, tar_frame, gact)
        group_dict = json.loads(row[3])
        person_num = len(group_dict)

        person_actions = []
        person_pixels = []
        # person_pixels = {pid: [] for pid in range(person_num)}

        int_pid = 0
        for pid in group_dict:
            person_actions.append(group_dict[pid][0])
            for i in range(1, NUM_FRAMES + 1):
                img_np = np.array(group_dict[pid][i])

                image = Image.fromarray(img_np.astype('uint8'), 'RGB')
                img_tensor = trans(image)

                person_pixels.append(img_tensor)
                # person_pixels[int_pid].append(img_tensor)
            int_pid += 1
        return person_pixels, person_actions, group_info

    def __len__(self):
        return self.tsv.num_rows()