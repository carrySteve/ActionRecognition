import json
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from io_common import FileProgressingbar, generate_lineidx
from tsv_io import TSVFile

TRAIN_PATH = "../feature/train.tsv"
VAL_PATH = "../feature/val.tsv"


class FeatureDataset(Dataset):
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

        # ginfo = row[0]
        glabel = row[0]
        actions = row[1][1:-1].split(',')
        actions = map(int, actions)
        features = json.loads(row[2])

        person_features = []
        for pid in features:
            person_feature = np.array(features[pid])
            person_features.append(person_feature)

        return person_features, actions, glabel
        # return person_features, actions, glabel, ginfo

    def __len__(self):
        return self.tsv.num_rows()
