# training script for neural nets
import argparse
import copy
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader_phase2 import FeatureDataset
from dataloader_phase2 import ReactiveGRU

# hyperparameters
__BATCH_SIZE = 1
__INPUT_DIM = 4096
__HIDDEN_DIM = 128
__ATTN_DIM = 64
__NUM_EPOCHS = 1000
__LEARNING_RATE = 0.0001
__GRAD_CLIP = 0.25
__FRAME_LEN = 10
__DROP_OUT = 0
# __DECAY_RATE = 0.0005
# "tanh" "relu" "nnTanh"
__GRU_ACTIVE = "tanh"
# __OLD_MODEL_PATH = './models/checkpoint_test_relu.pth'
__MODEL_PATH = './models/checkpoint_{}_{}_wd.pth'.format(
    __GRU_ACTIVE, __HIDDEN_DIM)
__LOG_PATH = './log/{}_{}_wd.txt'.format(__GRU_ACTIVE, __HIDDEN_DIM)
note = "dropout & L2 regularization"

PACTIONS = [
    'blocking', 'digging', 'falling', 'jumping', 'moving', 'setting',
    'spiking', 'standing', 'waiting'
]
__PACTION_NUM = 9

GACTIVITIES = [
    'r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass',
    'l_winpoint'
]
GACTIVITY_NUM = 8

device_ids = [0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

id2pact = {i: name for i, name in enumerate(PACTIONS)}
id2gact = {i: name for i, name in enumerate(GACTIVITIES)}


def collate_fn(batch):
    person_pixels, person_actions, group_info, ginfo = zip(*batch)

    return person_pixels[0], person_actions[0], group_info[0], ginfo[0]


def train_model(model, dataloaders_dict, criterion, optimizer):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with open(__LOG_PATH, 'a') as f:
        f.write('=================================================\n')
        f.write('hidden dimension:{} gru activate:{} {}\n'.format(
            __HIDDEN_DIM, __GRU_ACTIVE, note))
        f.write('=================================================\n')

    for epoch in range(__NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, __NUM_EPOCHS - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            length = 0.0

            # Iterate over data.
            for person_pixels, person_actions, group_info, ginfo in dataloaders_dict[
                    phase]:
                print(ginfo)
                person_num = len(person_actions)
                pidxes = range(person_num)
                # pidx: frame hiddens
                hidden_states = {
                    fidx: {pidx
                           for pidx in pidxes}
                    for fidx in range(__FRAME_LEN)
                }
                person_features = {fidx: 0.0 for fidx in range(__FRAME_LEN)}

                for fidx in range(__FRAME_LEN):
                    person_feature = []
                    # get all the people pixel in the frame
                    for pidx in range(person_num):
                        person_feature.append(person_pixels[pidx][fidx])
                    person_features[fidx] = torch.from_numpy(
                        np.array(person_feature)).float().to(device)

                # 1st iteration to get the hiddens
                for fidx in range(__FRAME_LEN):
                    nei_hiddens = []

                    if fidx == 0:
                        # initialization
                        prev_hidden = torch.zeros(person_num,
                                                  __HIDDEN_DIM).to(device)
                        nei_hiddens = torch.zeros(person_num, __HIDDEN_DIM,
                                                  person_num - 1).to(device)
                    else:
                        # get previous hidden
                        # also the temproal hidden at the first iteration
                        prev_hidden = hidden_states[fidx - 1]
                        # get the spatial hidden
                        for pidx in pidxes:
                            nei_idxes = range(person_num)
                            nei_idxes.remove(pidx)

                            # to store neighbors' hidden
                            temp_hidden = []
                            for nei_idx in nei_idxes:
                                temp_hidden.append(
                                    hidden_states[fidx - 1][nei_idx].view(
                                        1, -1))
                            nei_hiddens.append(
                                torch.cat(temp_hidden).to(device))

                        nei_hiddens = torch.stack(nei_hiddens).view(
                            person_num, __HIDDEN_DIM,
                            person_num - 1).to(device)
                        # -> (person_num, neighbor_num, hidden_dim)

                    spat_hidden = nei_hiddens

                    _, _, hidden = model(person_features[fidx], prev_hidden,
                                         spat_hidden, prev_hidden)

                    hidden.detach_()
                    # used to compute concatenate hiddens
                    hidden_states[fidx] = hidden

                # second iteration to train the model
                person_loss = 0.0
                prev_hidden = torch.zeros(person_num, __HIDDEN_DIM).to(device)
                pact_sum = torch.zeros(person_num, __PACTION_NUM).to(device)
                target_pact = torch.tensor(person_actions).to(device)
                for fidx in range(__FRAME_LEN):
                    nei_hiddens = []

                    if fidx == 0:
                        # initialization
                        nei_hiddens = torch.zeros(person_num, __HIDDEN_DIM,
                                                  person_num - 1).to(device)
                    else:
                        # get the spatial hidden
                        for pidx in pidxes:
                            nei_idxes = range(person_num)
                            nei_idxes.remove(pidx)

                            # to store neighbors' hidden
                            temp_hidden = []
                            for nei_idx in nei_idxes:
                                temp_hidden.append(
                                    hidden_states[fidx][nei_idx].view(1, -1))
                            nei_hiddens.append(
                                torch.cat(temp_hidden).to(device))

                        nei_hiddens = torch.stack(nei_hiddens).view(
                            person_num, __HIDDEN_DIM,
                            person_num - 1).to(device)

                    # get the temporal hidden
                    temp_hidden = hidden_states[fidx]
                    # cat the spatial hiddens
                    spat_hidden = nei_hiddens

                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        person_act, group_act, hidden = model(
                            person_features[fidx], temp_hidden, spat_hidden,
                            prev_hidden)

                        person_loss += criterion(person_act, target_pact)
                        pact_sum += person_act.detach_()

                        prev_hidden = hidden

                        if fidx == __FRAME_LEN - 1:
                            pact_sum /= __FRAME_LEN
                            pact_sum = F.softmax(pact_sum, dim=-1)

                            # group_loss = criterion(group_act,
                            #                         group_info)

                            _, preds = torch.max(pact_sum, 1)
                            # print(preds)
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                person_loss.backward()
                                optimizer.step()
                            # print(target_pact)

                            running_loss += person_loss.item() * person_num
                            # print(preds, target_pact.data)
                            running_corrects += torch.sum(
                                preds == target_pact.data)
                            length += person_num
                            # print(running_corrects, length)

            epoch_loss = running_loss / length
            epoch_acc = running_corrects.double() / length

            print('|{}|{}|'.format(
                time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))))
            print('{} {} Loss: {:.4f} Acc: {:.4f} '.format(
                epoch, phase, epoch_loss, epoch_acc))

            time_elapsed = time.time() - since

            with open(__LOG_PATH, 'a') as f:
                f.write('{}|{}  {} {} Loss: {:.4f} Acc: {:.4f}\n'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time())), epoch, phase,
                    epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.module.state_dict(), __MODEL_PATH)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open(__LOG_PATH, 'a') as f:
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        f.write('Best val Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


print('creating model')
model = ReactiveGRU(__INPUT_DIM, __PACTION_NUM, GACTIVITY_NUM, __HIDDEN_DIM,
                    __ATTN_DIM, __GRU_ACTIVE, __DROP_OUT).to(device)
# print('loading model from ' + __OLD_MODEL_PATH)
# model.load_state_dict(torch.load(__OLD_MODEL_PATH))

model = nn.DataParallel(model, device_ids=device_ids)
dataloaders_dict = {
    x: DataLoader(FeatureDataset(x),
                  batch_size=__BATCH_SIZE,
                  shuffle=True,
                  num_workers=0,
                  collate_fn=collate_fn)
    for x in ['train', 'val']
}
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model.parameters(), lr=__LEARNING_RATE)

# train_with_attn(model, dataloaders_dict, criterion, optimizer_ft)
model, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft)
