# training script for neural nets
import argparse
import copy
import math
import os
import sys
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from inception import inception_v3
# from model import ClassifierA
from torch.utils.data.dataloader import DataLoader
from image_loader import VolleyballDataset
import pickle

device_ids = [0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
__BATCH_SIZE = 1
__LEARNING_RATE = 1e-5
__WEIGHT_DECAY = 0.00004
__DROPOUT_RATIO = 0.2
__OUTPUT_HEIGHT = 87
__OUTPUT_WIDTH = 157
__CROP_SIZE = 5
__CHANNEL = 1056
__BBOX_NUM = 12
__CLASSIFIER_INPUT = __CHANNEL * __CROP_SIZE * __CROP_SIZE
__EMBED_DIM = 2048
__ACTION_WEIGHT = torch.tensor([1., 1., 2., 3., 1., 2., 2., 0.2,
                                1.]).to(device)
__ACTIONS_LOSS_WEIGHT = 0.5
__TIME_STEP = 12
__OLD_BACKBONE_PATH = './models/backbone/backbone 12-frames total_loss lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 epoch-38 dropout_ratio-0.2.pth'
__BONE_PATH = './models/backbone/'
__CLASSIFIER_PATH = './models/classifier/'

note = "feature extraction  \n\
        time: {}   \n\
        lr: {}  \n\
        weight_decay: {}    \n\
        crop_size: {}   \n\
        embed_dim: {}   \n\
        dropout_ratio:{}".format(
    time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())),
    __LEARNING_RATE, __WEIGHT_DECAY, __CROP_SIZE, __EMBED_DIM, __DROPOUT_RATIO)

__LOG_PATH = './log/B 8-frames lr-{} weight_decay-{} crop_size-{} embed_dim-{} drpout_ratio-{}.txt'.format(
    __LEARNING_RATE, __WEIGHT_DECAY, __CROP_SIZE, __EMBED_DIM, __DROPOUT_RATIO)

for path in [__BONE_PATH, __CLASSIFIER_PATH]:
    if (os.path.exists(path)):
        print('{} verified'.format(path))
    else:
        print('{} not exist'.format(path))
        sys.exit(0)

PACTIONS = [
    'blocking', 'digging', 'falling', 'jumping', 'moving', 'setting',
    'spiking', 'standing', 'waiting'
]
__PACTION_NUM = 9

GACTIVITIES = [
    'r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass',
    'l_winpoint'
]
__GACTIVITY_NUM = 8

id2pact = {i: name for i, name in enumerate(PACTIONS)}
id2gact = {i: name for i, name in enumerate(GACTIVITIES)}


def collate_fn(batch):
    img_tensor, bboxes, actions, activities, group_info = zip(*batch)
    return torch.cat(
        img_tensor, dim=0), np.stack(bboxes), np.vstack(actions), np.hstack(
            activities), np.hstack(group_info)


# def collate_fn(batch):
#     img_tensor, bboxes, actions, activities, group_info = zip(*batch)

#     return img_tensor[0], bboxes[0], actions[0], activities[0], group_info[0]


def train_model(backbone, dataloaders_dict, criterion_dict):
    since = time.time()

    with open(__LOG_PATH, 'a') as f:
        f.write('=================================================\n')
        f.write('{}\n'.format(note))
        f.write('=================================================\n')

    for phase in ['trainval', 'test']:
        total_action_loss = 0.0
        total_activity_loss = 0.0

        total_actions_accuracy = 0.0
        activities_accuracy = 0.0

        action_len = 0.0
        activity_len = 0.0
        total_len = 0.0

        backbone.eval()

        for img_tensor, bboxes, raw_actions, raw_activities, group_info in dataloaders_dict[
                phase]:
            action_loss = 0.0
            activity_loss = 0.0

            with torch.set_grad_enabled(False):

                mixed_5d, mixed_6e = backbone(img_tensor.to(device))
                # (BT, 288, 87, 157) (BT, 768, 43, 78)
                features_multiscale = []
                features_multiscale.append(mixed_5d)
                features_multiscale.append(
                    F.interpolate(
                        mixed_6e, (__OUTPUT_HEIGHT, __OUTPUT_WIDTH),
                        mode='bilinear',
                        align_corners=False))
                features_multiscale = torch.cat(
                    features_multiscale,
                    dim=1).view(__BATCH_SIZE, __TIME_STEP, __CHANNEL,
                                __OUTPUT_HEIGHT, __OUTPUT_WIDTH)
                # (B, T, 1056, 87, 157)

                # TODO: construct index on-the-fly
                boxes_features_multiscale = []
                for batch_idx in range(__BATCH_SIZE):
                    boxes_features_batch = []
                    feature = features_multiscale[batch_idx]
                    # (T, 1056, 87, 157)
                    batch_bboxes = bboxes[batch_idx]
                    # (T, N, 4)
                    print(batch_bboxes.shape)
                    batch_bboxes[:, :, [
                        0, 2
                    ]] = batch_bboxes[:, :, [0, 2]] * __OUTPUT_HEIGHT
                    # y, y+h * 87
                    batch_bboxes[:, :, [
                        1, 3
                    ]] = batch_bboxes[:, :, [1, 3]] * __OUTPUT_WIDTH
                    # x, x+2 * 157
                    for fidx in range(__TIME_STEP):
                        boxes_features_batch_frame = []
                        for pidx in range(__BBOX_NUM):
                            y0, x0, y1, x1 = map(
                                int, batch_bboxes[fidx][pidx].tolist())
                            crop_feature = feature[fidx, :, y0:(y1 + 1), x0:(
                                x1 + 1)].unsqueeze(dim=0)
                            # (1, 1056, H, W)
                            crop_resize_feature = F.interpolate(
                                crop_feature, (__CROP_SIZE, __CROP_SIZE),
                                mode='bilinear',
                                align_corners=False).squeeze()
                            # (1056, 5, 5)
                            boxes_features_batch_frame.append(
                                crop_resize_feature)
                        boxes_features_batch_frame = torch.stack(
                            boxes_features_batch_frame)
                        # (N, 1056, 5, 5)
                        boxes_features_batch.append(boxes_features_batch_frame)
                    boxes_features_batch = torch.stack(boxes_features_batch)
                    # [T, N, 1056, 5, 5]
                    boxes_features_multiscale.append(boxes_features_batch)
                boxes_features_multiscale = torch.stack(
                    boxes_features_multiscale)
                # [B, T, N, 1056, 5, 5]
                boxes_features_multiscale_flat = boxes_features_multiscale.view(
                    __BATCH_SIZE, __TIME_STEP, __BBOX_NUM, __CLASSIFIER_INPUT)
                # [B, T, N, 1056, 5, 5] -> [B, T, N, 26400]
                json_dict = {
                    'group_info':
                    group_info.tolist(),
                    'actions':
                    raw_actions.tolist(),
                    'activities':
                    raw_activities.tolist(),
                    'featuers':
                    boxes_features_multiscale_flat.cpu().numpy().tolist()
                }

                tsv_path = './data/feature 12-frames total_loss drop-{} {}.tsv'.format(
                    __DROPOUT_RATIO, phase)
                print('writing {} ...'.format(tsv_path))
                with open(tsv_path, 'a') as tsv_file:
                    tsv_file.write('{}\n'.format(json.dumps(json_dict)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def main():
    print('creating backbone')
    backbone = inception_v3().to(device)

    # Initialize the models for this run
    print('loading backbone from ' + __OLD_BACKBONE_PATH)
    backbone.load_state_dict(torch.load(__OLD_BACKBONE_PATH))
    dataloaders_dict = {
        x: DataLoader(
            VolleyballDataset(x),
            batch_size=__BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn)
        for x in ['trainval', 'test']
    }
    criterion_dict = {
        'action': nn.CrossEntropyLoss(weight=__ACTION_WEIGHT),
        'activity': nn.CrossEntropyLoss()
    }

    train_model(backbone, dataloaders_dict, criterion_dict)


if __name__ == "__main__":
    main()
