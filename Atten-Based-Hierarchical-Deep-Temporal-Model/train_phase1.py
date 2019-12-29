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

from dataloader_phase1 import VolleyballDataset
from models import ReactiveGRU
from backbone import vgg19_bn

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
# __OLD_MODEL_PATH = './models/checkpoint_tanh_128_vgg.pth'
__OLD_VGG_PATH = '../vgg/vgg_models/checkpoint_ex_119.pth'
__MODEL_PATH = './models/checkpoint_{}_{}_vgg.pth'.format(
    __GRU_ACTIVE, __HIDDEN_DIM)
__VGG_PATH = './vgg_models/checkpoint125.pth'

__LOG_PATH = './log/{}_{}_vgg.txt'.format(__GRU_ACTIVE, __HIDDEN_DIM)
note = "with vgg"

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

device_ids = [1]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

id2pact = {i: name for i, name in enumerate(PACTIONS)}
id2gact = {i: name for i, name in enumerate(GACTIVITIES)}


def collate_fn(batch):
    person_pixels, person_actions, group_info = zip(*batch)

    return person_pixels[0], person_actions[0], group_info[0]


def train_model(model, vgg, extractor, dataloaders_dict, criterion, optimizer):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_vgg_wts = copy.deepcopy(vgg.state_dict())
    extractor_dict = copy.deepcopy(extractor.state_dict())
    best_acc_lstm = 0.0
    best_acc_vgg = 0.0

    feature = {fidx: [] for fidx in range(__FRAME_LEN)}

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
                vgg.train()
            else:
                model.eval()  # Set model to evaluate mode
                vgg.eval()

            running_loss_vgg = 0.0
            running_corrects_vgg = 0.0
            length_vgg = 0.0

            running_loss_lstm = 0.0
            running_corrects_lstm = 0.0
            length_lstm = 0.0

            # Iterate over data.
            for inputs, person_actions, group_info in dataloaders_dict[phase]:
                print(group_info)
                # with open(__LOG_PATH, 'a') as f:
                #     f.write('ginfo: {}'.format(ginfo))
                person_num = len(person_actions)
                pidxes = range(person_num)

                features = {fidx: [] for fidx in range(__FRAME_LEN)}

                # fine tune vgg and extract features
                new_action = []
                for label in person_actions:
                    for i in range(__FRAME_LEN):
                        new_action.append(label)

                actions = [
                    new_action[i:i + __FRAME_LEN]
                    for i in range(0, len(new_action), __FRAME_LEN)
                ]

                for pidx in pidxes:
                    pixel = np.stack(inputs[pidx], axis=0)
                    action = np.array(actions[pidx])

                    pixel = torch.from_numpy(pixel).to(device)
                    # (batch_size, 3, 224, 224)

                    action = torch.from_numpy(action).to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # ============== fine tuning ================
                        outputs = vgg(pixel)
                        loss = criterion(outputs, action)

                        _, preds = torch.max(outputs, 1)  # (10,)

                    # === init extractor to get the extracted features ===
                    with torch.set_grad_enabled(False):
                        pretrained_dict = copy.deepcopy(vgg.state_dict())
                        # filter out unnecessary keys
                        pretrained_dict = {
                            k: v
                            for k, v in pretrained_dict.items()
                            if k in extractor_dict
                        }
                        # load the new state dict
                        extractor.load_state_dict(pretrained_dict)

                        feature = extractor(pixel)  # (10, 4096)

                        for fidx in range(__FRAME_LEN):
                            features[fidx].append(feature[fidx])

                    # statistics
                    running_loss_vgg += loss.item() * pixel.size(0)
                    running_corrects_vgg += torch.sum(preds == action.data)
                    length_vgg += pixel.size(0)

                # ========================== begin Reactive LSTM training ==========================
                hidden_states = {
                    fidx: {pidx
                           for pidx in pidxes}
                    for fidx in range(__FRAME_LEN)
                }
                person_features = {fidx: 0.0 for fidx in range(__FRAME_LEN)}

                for fidx in range(__FRAME_LEN):
                    person_features[fidx] = torch.stack(
                        features[fidx]).to(device)

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
                pact_vote = torch.zeros(person_num, __PACTION_NUM).to(device)
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
                        # person_act = F.softmax(person_act, dim=-1) # (person_num, __PACTION_NUM)

                        frame_prob, frame_pred = torch.max(person_act, 1)
                        frame_prob = frame_prob.cpu().tolist()
                        frame_pred = frame_pred.cpu().tolist()

                        for pidx, pact in enumerate(frame_pred):
                            pact_vote[pidx][pact] += frame_prob[pidx]

                        person_loss += criterion(person_act, target_pact)

                        prev_hidden = hidden

                        if fidx == __FRAME_LEN - 1:
                            # group_loss = criterion(group_act,
                            #                         group_info)

                            _, preds = torch.max(pact_vote, 1)
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                person_loss.backward()
                                optimizer.step()
                            # print(target_pact)

                            running_loss_lstm += person_loss.item(
                            ) * person_num
                            # print(preds, target_pact.data)
                            running_corrects_lstm += torch.sum(
                                preds == target_pact.data)
                            length_lstm += person_num
                            # print(running_corrects, length)

            epoch_loss_vgg = running_loss_vgg / length_vgg
            epoch_acc_vgg = running_corrects_vgg.double() / length_vgg
            epoch_loss_lstm = running_loss_lstm / length_lstm
            epoch_acc_lstm = running_corrects_lstm.double() / length_lstm

            print('|{}|{}|'.format(
                time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))))
            print('{} {} VGG Loss: {:.4f} Acc: {:.4f} '.format(
                epoch, phase, epoch_loss_vgg, epoch_acc_vgg))
            print('{} {} LSTM Loss: {:.4f} Acc: {:.4f} \n'.format(
                epoch, phase, epoch_loss_lstm, epoch_acc_lstm))

            with open(__LOG_PATH, 'a') as f:
                f.write('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                f.write('{} {} VGG Loss: {:.4f} Acc: {:.4f} '.format(
                    epoch, phase, epoch_loss_vgg, epoch_acc_vgg))
                f.write('{} {} LSTM Loss: {:.4f} Acc: {:.4f} \n'.format(
                    epoch, phase, epoch_loss_lstm, epoch_acc_lstm))

            # deep copy the model
            if phase == 'val' and epoch_acc_vgg > best_acc_vgg:
                best_acc_vgg = epoch_acc_vgg
                best_vgg_wts = copy.deepcopy(vgg.state_dict())
                torch.save(vgg.module.state_dict(), __VGG_PATH)
            if phase == 'val' and epoch_acc_lstm > best_acc_lstm:
                best_acc_vgg_lstm = epoch_acc_lstm
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.module.state_dict(), __MODEL_PATH)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} {:4f}'.format(best_acc_vgg, best_acc_lstm))
    with open(__LOG_PATH, 'a') as f:
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        f.write('Best val Acc: {:4f} {:4f}\n'.format(best_acc_vgg,
                                                     best_acc_lstm))


def main():
    print('creating ReactiveGRU')
    model = ReactiveGRU(__INPUT_DIM, __PACTION_NUM, GACTIVITY_NUM,
                        __HIDDEN_DIM, __ATTN_DIM, __GRU_ACTIVE,
                        __DROP_OUT).to(device)
    print('creating VGG')
    vgg = vgg19_bn(pretrained=True).to(device)
    print('creating extractor')
    extractor = vgg19_bn(pretrained=True).to(device)
    extractor.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])

    # model.load_state_dict(torch.load(__OLD_MODEL_PATH))

    # Initialize the model for this run
    # print('loading model from ' + __OLD_MODEL_PATH)
    # model.load_state_dict(torch.load(__OLD_MODEL_PATH))
    print('loading vgg from ' + __OLD_VGG_PATH)
    vgg.load_state_dict(torch.load(__OLD_VGG_PATH))

    model = nn.DataParallel(model, device_ids=device_ids)
    vgg = nn.DataParallel(vgg, device_ids=device_ids)
    extractor = nn.DataParallel(extractor, device_ids=device_ids)

    dataloaders_dict = {
        x: DataLoader(VolleyballDataset(x),
                      batch_size=__BATCH_SIZE,
                      shuffle=True,
                      num_workers=0,
                      collate_fn=collate_fn)
        for x in ['train', 'val']
    }
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=__LEARNING_RATE)

    train_model(model, vgg, extractor, dataloaders_dict, criterion,
                optimizer_ft)


if __name__ == "__main__":
    main()
