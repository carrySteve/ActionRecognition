# -*- coding: utf-8 -*-

import json
import sys
import os
import gc
import psutil
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

TRAINVAL_PATH = './data/trainval_a.tsv'
TEST_PATH = './data/test_a.tsv'

__RESIZE_WIDTH = 1280
__RESIZE_HEIGHT = 720

test_transform = transforms.Compose([
    transforms.Resize((__RESIZE_HEIGHT, __RESIZE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class VolleyballDataset(Dataset):
    def __init__(self, phase):
        self.phase = phase
        if phase == "trainval":
            self.tsv_path = TRAINVAL_PATH
        else:
            self.tsv_path = TEST_PATH

        self.tsv = TSVFile(self.tsv_path)

    def __getitem__(self, idx):
        num_boxes = 12
        row = self.tsv.seek(idx)
        json_dict = json.loads(row[0])

        set_id = json_dict['video']
        tar_frame = json_dict['tf']
        group_activities = json_dict['glabel']
        group_info = np.array((set_id, tar_frame, group_activities))
        person_actions = np.array(json_dict['plabel'])
        person_num = person_actions.shape[0]
        bboxes = np.array(json_dict['bboxes'])
        if person_num != num_boxes:
            bboxes = np.vstack([bboxes, bboxes[:num_boxes - len(bboxes[-1])]])
            person_actions = np.hstack([person_actions, person_actions[:num_boxes - person_num]])
        img_np = np.array(json_dict['pixels'])

        image = Image.fromarray(img_np.astype('uint8'), 'RGB')
        img_tensor = test_transform(image)

        return img_tensor, bboxes, person_actions, np.array(group_activities), group_info

    def __len__(self):
        return self.tsv.num_rows()


def collate_fn(batch):
    img_tensor, bboxes, actions, activities, group_info = zip(*batch)
    # groups_feature = torch.cat(img_tensor, dim=1)

    return torch.stack(img_tensor), bboxes, np.hstack(actions), np.vstack(
        activities), np.hstack(group_info)

dataloaders_dict = {
    x: torch.utils.data.DataLoader(
        VolleyballDataset(x),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)
    for x in ['trainval']
}

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


for img_tensor, bboxes, actions, activities, group_info in dataloaders_dict['trainval']:
    
    cpuStats()
    memReport() 
    # print(sys.getsizeof(img_tensor))
    # print(bboxes.shape)
    # print(actions.shape)
    # print(actions)
    # print(group_info)
    # print(group_info)
    sys.exit(0)
