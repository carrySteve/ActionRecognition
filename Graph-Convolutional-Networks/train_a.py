# training script for neural nets
import argparse
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.inception import inception_v3
from models.classifiers import ClassifierA
from torch.utils.data.dataloader import DataLoader
from image_loader_a import VolleyballDataset
import pickle

device_ids = [0, 1]
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
# hyperparameters
__BATCH_SIZE = 8
__NUM_EPOCHS = 100
__LEARNING_RATE = 1e-5
__WEIGHT_DECAY = 0.00004
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
__DROPOUT_RATIO = 0.2
__TEST_GAP = 1
__STORE_GAP = 1
# __OLD_BACKBONE_PATH = './models/backbone/inception_v3.pth'
__OLD_BACKBONE_PATH = './models/backbone/backbone 8-frames total_loss lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 epoch-5 dropout_ratio-0.2.pth'
__OLD_CLASSIFIER_PATH = './models/classifier/classifierA 8-frames total_loss lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 epoch-5 dropout_ratio-0.2.pth'
__BONE_PATH = './models/backbone/frames/total_loss/'
__CLASSIFIER_PATH = './models/classifier/frames/total_loss/'

note = "implement social scene  \n\
        time: {}   \n\
        lr: {}  \n\
        weight_decay: {}    \n\
        crop_size: {}   \n\
        embed_dim: {}   \n\
        dropout_ratio:{} \n".format(
    time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())),
    __LEARNING_RATE, __WEIGHT_DECAY, __CROP_SIZE, __EMBED_DIM, __DROPOUT_RATIO)

__LOG_PATH = './log/A 12-frames total_loss lr-{} weight_decay-{} crop_size-{} embed_dim-{} drpout_ratio-{}.txt'.format(
    __LEARNING_RATE, __WEIGHT_DECAY, __CROP_SIZE, __EMBED_DIM, __DROPOUT_RATIO)

for path in [__BONE_PATH, __CLASSIFIER_PATH]:
    if (os.path.exists(path)):
        print('{} verified'.format(path))
    else:
        os.makedirs(path)
        print('making {}'.format(path))

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

# def collate_fn(batch):
#     img_tensor, bboxes, actions, activities, group_info = zip(*batch)

#     return torch.stack(img_tensor), bboxes, np.hstack(actions), np.hstack(
#         activities), np.hstack(group_info)


def collate_fn(batch):
    img_tensor, bboxes, actions, activities, group_info = zip(*batch)
    return torch.cat(img_tensor, dim=0), np.stack(bboxes), torch.cat(
        actions, dim=0), np.stack(activities), np.vstack(group_info)


def train_model(backbone, classifier, dataloaders_dict, criterion_dict,
                optimizer):
    since = time.time()

    with open(__LOG_PATH, 'a') as f:
        f.write('=================================================\n')
        f.write('{}\n'.format(note))
        f.write('=================================================\n')

    for epoch in range(__NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, __NUM_EPOCHS))
        print('-' * 10)

        # if epoch == 38:
        #     phases = ['test']
        # else:
        phases = ['trainval', 'test']

        for phase in phases:
            total_action_loss = 0.0
            total_activity_loss = 0.0

            total_actions_accuracy = 0.0
            activities_accuracy = 0.0

            action_len = 0.0
            activity_len = 0.0
            total_len = 0.0
            if phase == 'trainval':
                backbone.train()
                classifier.train()
            else:
                backbone.eval()
                classifier.eval()

            for img_tensor, bboxes, actions, activities, group_info in dataloaders_dict[
                    phase]:
                action_loss = 0.0
                activity_loss = 0.0

                with torch.set_grad_enabled(phase == 'trainval'):
                    # actions = torch.from_numpy(actions).to(device)
                    actions = actions.to(device)
                    activities = torch.from_numpy(activities).to(device)
                    batch_size = activities.shape[0]

                    mixed_5d, mixed_6e = backbone(img_tensor.to(device))
                    # (1, 288, 87, 157) (1, 768, 43, 78)
                    features_multiscale = []
                    features_multiscale.append(mixed_5d)
                    # print(mixed_5d.permute(0, 2, 3, 1))
                    features_multiscale.append(
                        F.interpolate(mixed_6e,
                                      (__OUTPUT_HEIGHT, __OUTPUT_WIDTH),
                                      mode='bilinear',
                                      align_corners=False))
                    features_multiscale = torch.cat(features_multiscale, dim=1)
                    # (1, 1056, 87, 157)

                    # [B, N, 4] -> [BN, 4]
                    # TODO: construct index on-the-fly
                    boxes_features_multiscale = []
                    for batch_idx in range(batch_size):
                        boxes_features_batch = []
                        feature = features_multiscale[batch_idx].unsqueeze(
                            dim=0)
                        # (1, 1056, 87, 157)
                        batch_bboxes = bboxes[batch_idx]
                        # (N, 4)
                        batch_bboxes[:, [
                            0, 2
                        ]] = batch_bboxes[:, [0, 2]] * __OUTPUT_HEIGHT
                        # y, y+h * 87
                        batch_bboxes[:, [
                            1, 3
                        ]] = batch_bboxes[:, [1, 3]] * __OUTPUT_WIDTH
                        # x, x+2 * 157
                        for pidx in range(__BBOX_NUM):
                            y0, x0, y1, x1 = map(int,
                                                 batch_bboxes[pidx].tolist())
                            crop_feature = feature[:, :, y0:(y1 + 1), x0:(x1 +
                                                                          1)]
                            crop_resize_feature = F.interpolate(
                                crop_feature, (__CROP_SIZE, __CROP_SIZE),
                                mode='bilinear',
                                align_corners=False).squeeze()
                            boxes_features_batch.append(crop_resize_feature)
                        boxes_features_batch = torch.stack(
                            boxes_features_batch)
                        # [N, 1056, 5, 5]
                        boxes_features_multiscale.append(boxes_features_batch)
                    boxes_features_multiscale = torch.stack(
                        boxes_features_multiscale)
                    # [B, N, 1056, 5, 5]

                    boxes_features_multiscale_flat = boxes_features_multiscale.view(
                        batch_size, __BBOX_NUM, __CLASSIFIER_INPUT)
                    # [B, N, 1056, 5, 5] -> [B, N, 26400]

                    action_logits, activity_logits = classifier(
                        boxes_features_multiscale_flat)

                    action_loss = criterion_dict['action'](
                        action_logits, actions) / (batch_size * __BBOX_NUM)
                    activity_loss = criterion_dict['activity'](
                        activity_logits, activities) / batch_size

                    total_loss = __ACTIONS_LOSS_WEIGHT * action_loss + activity_loss

                    if phase == 'trainval':
                        optimizer.zero_grad()
                        # action_loss.backward()
                        total_loss.backward()
                        optimizer.step()

                    _, actions_labels = torch.max(action_logits, 1)
                    _, activities_labels = torch.max(activity_logits, 1)

                    total_action_loss += action_loss
                    total_activity_loss += activity_loss

                    total_actions_accuracy += torch.sum(
                        actions_labels == actions.data)
                    activities_accuracy += torch.sum(
                        activities_labels == activities.data)

                    action_len += batch_size * __BBOX_NUM
                    activity_len += batch_size
                    total_len += 1

                    print('{} {} Person Loss: {:.4f} Acc: {:.4f} \n'.format(
                        epoch + 1, phase, total_action_loss / total_len,
                        total_actions_accuracy.double() / action_len))

                    print('{} {} Group Loss: {:.4f} Acc: {:.4f} \n'.format(
                        epoch + 1, phase, total_activity_loss / total_len,
                        activities_accuracy.double() / activity_len))

            epoch_action_loss = total_action_loss / total_len
            epoch_action_acc = total_actions_accuracy.double() / action_len

            epoch_activity_loss = total_activity_loss / total_len
            epoch_activity_acc = activities_accuracy.double() / activity_len

            print('|{}|{}|'.format(
                time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))))
            print('{} {} Person Loss: {:.4f} Acc: {:.4f} \n'.format(
                epoch + 1, phase, epoch_action_loss, epoch_action_acc))
            print('{} {} Group Loss: {:.4f} Acc: {:.4f} \n'.format(
                epoch + 1, phase, epoch_activity_loss, epoch_activity_acc))

            with open(__LOG_PATH, 'a') as f:
                f.write('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                f.write('{} {} Person Loss: {:.4f} Acc: {:.4f} '.format(
                    epoch + 1, phase, epoch_action_loss, epoch_action_acc))
                f.write('{} {} Group Loss: {:.4f} Acc: {:.4f} \n'.format(
                    epoch + 1, phase, epoch_activity_loss, epoch_activity_acc))

            if (epoch + 1) % __STORE_GAP == 0:
                torch.save(
                    backbone.module.state_dict(), __BONE_PATH +
                    'backbone 12-frames total_loss lr-{} weight_decay-{} crop_size-{} embed_dim-{} epoch-{} dropout_ratio-{}.pth'
                    .format(__LEARNING_RATE, __WEIGHT_DECAY, __CROP_SIZE,
                            __EMBED_DIM, epoch + 1, __DROPOUT_RATIO))
                torch.save(
                    classifier.module.state_dict(), __CLASSIFIER_PATH +
                    'classifierA 12-frames total_loss lr-{} weight_decay-{} crop_size-{} embed_dim-{} epoch-{} dropout_ratio-{}.pth'
                    .format(__LEARNING_RATE, __WEIGHT_DECAY, __CROP_SIZE,
                            __EMBED_DIM, epoch + 1, __DROPOUT_RATIO))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    with open(__LOG_PATH, 'a') as f:
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))


def main():
    print('creating backbone')
    backbone = inception_v3().to(device)

    print('creating classifier')
    classifier = ClassifierA(feature_dim=__CLASSIFIER_INPUT,
                             embed_dim=__EMBED_DIM,
                             dropout_ratio=__DROPOUT_RATIO).to(device)

    # Initialize the models for this run
    # backbone_dict = backbone.state_dict()
    # pretrained_dict = torch.load(__OLD_BACKBONE_PATH)
    # pretrained_dict = {
    #     k: v
    #     for k, v in pretrained_dict.items() if k in backbone_dict
    # }
    # print('loading backbone keys {} from {}'.format(pretrained_dict.keys(),
    #                                                 __OLD_BACKBONE_PATH))
    # backbone_dict.update(pretrained_dict)
    # backbone.load_state_dict(backbone_dict)

    print('loading backbone from ' + __OLD_BACKBONE_PATH)
    backbone.load_state_dict(torch.load(__OLD_BACKBONE_PATH))
    print('loading classifier from ' + __OLD_CLASSIFIER_PATH)
    classifier.load_state_dict(torch.load(__OLD_CLASSIFIER_PATH))

    backbone = nn.DataParallel(backbone, device_ids=device_ids)
    classifier = nn.DataParallel(classifier, device_ids=device_ids)

    dataloaders_dict = {
        x: DataLoader(VolleyballDataset(x),
                      batch_size=__BATCH_SIZE,
                      shuffle=True,
                      num_workers=2,
                      collate_fn=collate_fn)
        for x in ['trainval', 'test']
    }
    criterion_dict = {
        'action': nn.CrossEntropyLoss(weight=__ACTION_WEIGHT),
        'activity': nn.CrossEntropyLoss()
    }

    params = list(backbone.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params,
                           lr=__LEARNING_RATE,
                           weight_decay=__WEIGHT_DECAY)

    train_model(backbone, classifier, dataloaders_dict, criterion_dict,
                optimizer)


if __name__ == "__main__":
    main()
