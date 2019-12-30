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

TRAINVAL_PATH = '../data/trainval_info_12frames.tsv'
TEST_PATH = '../data/test_info_12frames.tsv'

__RESIZE_WIDTH = 1280
__RESIZE_HEIGHT = 720
HIGH_RESOLUTION = [2, 37, 38, 39, 40, 41, 44, 45]

resized_transform = transforms.Compose([
    transforms.Resize((__RESIZE_HEIGHT, __RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class VolleyballDataset(Dataset):
    def __init__(self, phase):
        self.phase = phase
        self.image_path = "../dataset/Volleyball/videos/"
        if phase == "trainval":
            self.tsv_path = TRAINVAL_PATH
        else:
            self.tsv_path = TEST_PATH

        self.tsv = TSVFile(self.tsv_path)

    def __getitem__(self, idx):
        num_boxes = 12
        row = self.tsv.seek(idx)
        json_dict = json.loads(row[0])

        set_idx = json_dict['video']
        target_fidx = int(json_dict['target'])
        frame_idxes = int(json_dict['frame'])

        group_activities = np.array(json_dict['glabel'])
        group_info = np.array((set_idx, target_fidx, frame_idxes))

        person_actions = np.array(json_dict['plabel'])
        person_num = person_actions.shape[0]
        if person_num != num_boxes:
            person_actions = np.hstack(
                [person_actions, person_actions[:num_boxes - person_num]])

        person_actions = torch.tensor(person_actions)

        bboxes = np.array(json_dict['bboxes'])
        bboxes_total = np.concatenate(
            [bboxes, bboxes[:num_boxes - person_num]], axis=0)

        img_tensor = []
        img_path = self.image_path + '{}/{}/{}.jpg'.format(
            set_idx, target_fidx, frame_idxes)
        frame_img = Image.open(img_path)
        if set_idx in HIGH_RESOLUTION:
            img_tensor.append(resized_transform(frame_img))
        else:
            img_tensor.append(transform(frame_img))

        # (B, 3, 780, 1280)
        return torch.stack(
            img_tensor
        ), bboxes_total, person_actions, group_activities, group_info

    def __len__(self):
        return self.tsv.num_rows()