# training script for neural nets
import argparse
import copy
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
# TODO:
from torch.utils.data.dataloader import DataLoader
from feature_loader_b import VolleyballDataset
import pickle

device_ids = [0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
__BATCH_SIZE = 16
__NUM_EPOCHS = 1000
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
__TEST_GAP = 5
__TEST_AFTER = 50
__STORE_GAP = 1

__NUM_WORKERS = 2
__ATTEN_METHOD = 'normal'
# 'normal' 'general' 'concat'
__TIME_STEP = 10
__LOSS_MODE = 'person'
__CLASSIFIER_MODE = 'gcn'
__FRAME_MODE = 'frames'
__GROUP_POOL = 'max'
__RNN = 'gru'

__LEARNING_RATE = 1e-3
__MOMENTUM = 0.9
__WEIGHT_DECAY = 4e-5

if __CLASSIFIER_MODE == 'original':
    print('from model import ClassifierB')
    from models.classifiers import ClassifierB
elif __CLASSIFIER_MODE == 'gcn':
    print('from model_b_gcn import ClassifierB')
    from models.model_b_gcn import ClassifierB
elif __CLASSIFIER_MODE == 'gcn-gru':
    print('from model_b_res_gru import ClassifierB')
    from models.model_b_res_gru import ClassifierB
elif __CLASSIFIER_MODE == 'multi':
    print('from model_b_multi import ClassifierB')
    from models.model_b_multihead import ClassifierB
else:
    print('wrong classifier B mode')
    sys.exit(0)

__INFO = 'SGD lr-{} momentum-{} weight_decay-{} batch_size-{} atten-{} time_step-{} rnn-{} loss-{} frame_mode-{} mode-{} group_pool-{} crop_size-{} embed_dim-{} drpout_ratio-{}'.format(
    __LEARNING_RATE, __MOMENTUM, __WEIGHT_DECAY, __BATCH_SIZE, __ATTEN_METHOD,
    __TIME_STEP, __RNN, __LOSS_MODE, __FRAME_MODE, __CLASSIFIER_MODE,
    __GROUP_POOL, __CROP_SIZE, __EMBED_DIM, __DROPOUT_RATIO)

# __OLD_CLASSIFIERA_PATH = './models/classifier/classifierA 8-frames total_loss lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 epoch-57 dropout_ratio-0.2.pth'

__OLD_CLASSIFIERA_PATH = './models/classifier/classifierA frames lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 epoch-43 dropout_ratio-0.2.pth'

# __OLD_CLASSIFIERA_PATH = './models/classifier/classifierA 12-frames total_loss lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 epoch-38 dropout_ratio-0.2.pth'

__CLASSIFIER_PATH = './models/classifier/{}/{}/{}/{}/'.format(
    __FRAME_MODE, __LOSS_MODE, __CLASSIFIER_MODE, __GROUP_POOL)
# __OLD_CLASSIFIERB_PATH = './models/classifier/classifierB1 lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 dropout_ratio-0.2 epoch-33.pth'
# __OLD_CLASSIFIERB_PATH = './models/classifier/frames/person/gcn/max/classifierB batch_size-16 atten_method-normal time_step-10 loss-person frame_mode-frames mode-gcn group_pool-max lr-1e-05 weight_decay-4e-05 crop_size-5 embed_dim-2048 drpout_ratio-0.2 epoch-6.pth'
__COUNTINUE_EPOCH = 171
__OLD_CLASSIFIERB_PATH = './models/classifier/frames/person/gcn/max/classifierB SGD lr-{} momentum-0.9 weight_decay-4e-05 batch_size-16 atten-normal time_step-10 rnn-gru loss-person frame_mode-frames mode-gcn group_pool-max crop_size-5 embed_dim-2048 drpout_ratio-0.2 epoch-{}.pth'.format(
    __LEARNING_RATE, __COUNTINUE_EPOCH)

for path in [__CLASSIFIER_PATH]:
    if (os.path.exists(path)):
        print('{} verified'.format(path))
    else:
        os.makedirs(path)
        print('making {}'.format(path))

note = "time: {}\n{} ".format(
    time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())),
    __INFO.replace(' ', '\n'))

__LOG_PATH = './log/B {}.txt'.format(__INFO)

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
#     group_info, actions, activities, features = zip(*batch)
#     return torch.cat(
#         features, dim=0), np.concatenate(
#             actions, axis=0), np.concatenate(
#                 activities, axis=0), torch.cat(
#                     group_info, dim=0)


def collate_fn(batch):
    group_info, actions, activities, features = zip(*batch)
    return torch.cat(
        features,
        dim=0), torch.stack(actions), torch.stack(activities), torch.cat(
            group_info, dim=0)


def train_model(classifierB, dataloaders_dict, criterion_dict, optimizer):
    since = time.time()

    with open(__LOG_PATH, 'a') as f:
        f.write('=================================================\n')
        f.write('{}\n'.format(note))
        f.write('=================================================\n')

    for epoch in range(__COUNTINUE_EPOCH, __NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, __NUM_EPOCHS))
        print('-' * 10)

        # # TODO:
        # if (epoch + 1) % __TEST_GAP != 0 or epoch <= __TEST_AFTER:
        phases = ['trainval']
        # else:
        #     phases = ['trainval', 'test']

        for phase in phases:
            total_action_loss = 0.0
            total_activity_loss = 0.0

            total_actions_accuracy = 0.0
            total_actions_target_accuracy = 0.0
            total_activities_accuracy = 0.0
            total_activity_target_accuracy = 0.0

            action_len = 0.0
            target_action_len = 0.0
            activity_len = 0.0
            target_activity_len = 0.0
            total_len = 0.0
            if phase == 'trainval':
                classifierB.train()
            else:
                classifierB.eval()

            for features, actions, activities, group_info in dataloaders_dict[
                    phase]:
                action_loss = 0.0
                activity_loss = 0.0

                with torch.set_grad_enabled(phase == 'trainval'):
                    batch_size = activities.shape[0]
                    # actions = torch.tensor(actions).to(device)
                    features = features.to(device)
                    actions = actions.to(device)
                    activities = activities.to(device)
                    action_logits, activity_logits = classifierB(features)
                    # [B, T, N, 9] [B, T, 8]

                    soft = nn.Softmax(dim=-1)
                    actions_preds = soft(action_logits)
                    actions_preds_target = torch.mean(actions_preds, dim=1)
                    # [B, N, 9]
                    # print(actions_preds_target.shape)
                    _, actions_labels_target = torch.max(
                        actions_preds_target, dim=2)
                    total_actions_target_accuracy += torch.sum(
                        actions_labels_target ==
                        actions[:, :__BBOX_NUM].data).item()
                    # activity predicts with T average
                    activity_preds = soft(activity_logits)
                    activity_preds_target = torch.mean(activity_preds, dim=1)
                    # [B, 8]
                    _, activity_labels_target = torch.max(
                        activity_preds_target, dim=1)
                    total_activity_target_accuracy += torch.sum(
                        activity_labels_target ==
                        activities[:, 0].data).item()

                    actions = actions.view(-1)
                    action_logits = action_logits.view(-1, __PACTION_NUM)

                    activities = activities.view(-1)
                    activity_logits = activity_logits.view(-1, __GACTIVITY_NUM)

                    action_loss = criterion_dict['action'](
                        action_logits,
                        actions) / (batch_size * __TIME_STEP * __BBOX_NUM)
                    activity_loss = criterion_dict['activity'](
                        activity_logits,
                        activities) / (batch_size * __TIME_STEP)

                    total_loss = __ACTIONS_LOSS_WEIGHT * action_loss + activity_loss

                    if phase == 'trainval':
                        optimizer.zero_grad()
                        # TODO
                        if __LOSS_MODE == 'person':
                            action_loss.backward()
                        elif __LOSS_MODE == 'group':
                            activity_loss.backward()
                        elif __LOSS_MODE == 'total':
                            total_loss.backward()
                        else:
                            print('wrong loss')
                            sys.exit(0)
                        optimizer.step()

                    _, actions_labels = torch.max(action_logits, 1)
                    _, activities_labels = torch.max(activity_logits, 1)

                    total_action_loss += action_loss.item()
                    total_activity_loss += activity_loss.item()

                    total_actions_accuracy += torch.sum(
                        actions_labels == actions.data).item()
                    total_activities_accuracy += torch.sum(
                        activities_labels == activities.data).item()

                    action_len += batch_size * __TIME_STEP * __BBOX_NUM
                    target_action_len += batch_size * __BBOX_NUM
                    activity_len += batch_size * __TIME_STEP
                    target_activity_len += batch_size
                    total_len += 1

                    print('{} {} Person Loss: {} Acc: {}% Target Acc: {}%'.
                          format(
                              epoch + 1, phase, total_action_loss / total_len,
                              total_actions_accuracy * 100 / action_len,
                              total_actions_target_accuracy * 100 /
                              target_action_len))

                    print(
                        '{} {} Group Loss: {} Acc: {}% Target Acc: {}%'.format(
                            epoch + 1, phase, total_activity_loss / total_len,
                            total_activities_accuracy * 100 / activity_len,
                            total_activity_target_accuracy * 100 /
                            target_activity_len))

            epoch_action_loss = total_action_loss / total_len
            epoch_action_acc = total_actions_accuracy / action_len

            epoch_activity_loss = total_activity_loss / total_len
            epoch_activity_acc = total_activities_accuracy / activity_len

            print('|{}|{}|'.format(
                time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))))
            print('{} {} Person Loss: {} Acc: {}% Target Acc: {}%'.format(
                epoch + 1, phase, total_action_loss / total_len,
                total_actions_accuracy * 100 / action_len,
                total_actions_target_accuracy * 100 / target_action_len))
            print('{} {} Group Loss: {} Acc: {}% Target Acc: {}%'.format(
                epoch + 1, phase, total_activity_loss / total_len,
                total_activities_accuracy * 100 / activity_len,
                total_activity_target_accuracy * 100 / target_activity_len))

            with open(__LOG_PATH, 'a') as f:
                f.write('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                f.write(
                    '{} {} Person Loss: {} Acc: {}% Target Acc: {}%\n'.format(
                        epoch + 1, phase, total_action_loss / total_len,
                        total_actions_accuracy * 100 / action_len,
                        total_actions_target_accuracy * 100 /
                        target_action_len))
                f.write('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                f.write(
                    '{} {} Group Loss: {} Acc: {}% Target Acc: {}% \n'.format(
                        epoch + 1, phase, total_activity_loss / total_len,
                        total_activities_accuracy * 100 / activity_len,
                        total_activity_target_accuracy * 100 /
                        target_activity_len))

            if (epoch + 1) % __STORE_GAP == 0:
                torch.save(
                    classifierB.module.state_dict(),
                    __CLASSIFIER_PATH + 'classifierB {} epoch-{}.pth'.format(
                        __INFO, epoch + 1))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    with open(__LOG_PATH, 'a') as f:
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))


def main():
    print('creating classifierB')
    classifierB = ClassifierB(
        feature_dim=__CLASSIFIER_INPUT,
        embed_dim=__EMBED_DIM,
        dropout_ratio=__DROPOUT_RATIO,
        group_pool=__GROUP_POOL).to(device)

    # pretrained_dict = torch.load(__OLD_CLASSIFIERA_PATH)
    # classifierB_dict = classifierB.state_dict()

    # pretrained_dict = {
    #     k: v
    #     for k, v in pretrained_dict.items()
    #     if k in ['embed_layer.0.weight', 'embed_layer.0.bias']
    # }
    # classifierB_dict.update(pretrained_dict)
    # print('classifierB loading {}\nfrom {}'.format(pretrained_dict.keys(),
    #                                                __OLD_CLASSIFIERA_PATH))
    # classifierB.load_state_dict(classifierB_dict)

    classifierB.load_state_dict(torch.load(__OLD_CLASSIFIERB_PATH))

    classifierB = nn.DataParallel(classifierB, device_ids=device_ids)

    dataloaders_dict = {
        x: DataLoader(
            VolleyballDataset(x, __FRAME_MODE),
            batch_size=__BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn)
        for x in ['trainval', 'test']
        # for x in ['trainval']
    }
    criterion_dict = {
        'action': nn.CrossEntropyLoss(weight=__ACTION_WEIGHT),
        'activity': nn.CrossEntropyLoss()
    }

    optimizer = optim.SGD(
        classifierB.parameters(),
        lr=__LEARNING_RATE,
        momentum=__MOMENTUM,
        weight_decay=__WEIGHT_DECAY)

    train_model(classifierB, dataloaders_dict, criterion_dict, optimizer)


if __name__ == "__main__":
    main()
