# -*- coding: utf-8 -*-

import json
import sys
import math
import random
from itertools import chain

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from io_common import FileProgressingbar, generate_lineidx, img_from_base64
from tsv_io import TSVFile

TRAINVAL_PATH = './data/feature drop-0.2 trainval.tsv'
TEST_PATH = './data/feature drop-0.2 test.tsv'
NUM_FRAMES = 10

class VolleyballDataset(Dataset):
    def __init__(self, phase):
        self.phase = phase
        if phase == "trainval":
            self.tsv_path = TRAINVAL_PATH
        else:
            self.tsv_path = TEST_PATH

        self.tsv = TSVFile(self.tsv_path)

    def __getitem__(self, idx):
        row = self.tsv.seek(idx)
        json_dict = json.loads(row[0])

        group_info = torch.tensor(json_dict['group_info'])
        actions = torch.tensor(json_dict['actions']).repeat(NUM_FRAMES)
        activities = np.array(json_dict['activities']).repeat(NUM_FRAMES)
        features = torch.tensor(json_dict['featuers'])

        return group_info, actions, activities, features

    def __len__(self):
        return self.tsv.num_rows()