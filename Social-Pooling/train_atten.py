# training script for neural nets
import torch
from torch.utils.data import DataLoader
from train_dataset import TrainDataset
from models import AttnGRU

import argparse
import matplotlib.pyplot as plt

import sys

_id = 0
_position = 1
_class = 2

# hyperparameters
__INPUT_DIM = 2
__OUTPUT_DIM = 2
__HIDDEN_DIM = 128
__ATTN_DIM = 64
__NUM_EPOCHS = 100
__LEARNING_RATE = 0.003
__NUM_SCENES = 4
__GRAD_CLIP = 0.25
__OBSERVE_LEN = 18
__FRAME_GAP = 15

classes = ["Pedestrian", "Biker", "Skater", "Cart"]

loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)


def collate_fn(data):
    x, y = zip(*data)
    return x, y


def step_through_scene(models, scene, epoch, num_epochs, s, calculate_loss):
    outlay_dict = scene[0]
    class_dict = scene[1]
    path_dict = scene[2]
    frames = outlay_dict.keys()
    frames = sorted(frames)
    # prev_hiddens = {}
    hidden_states = {}
    occu_dict = {}

    # image drawing
    real_x = {}
    real_y = {}
    predict_x = {}
    predict_y = {}

    cost = {c: [] for c in classes}

    for frame in frames:
        print("EPOCH {} / {} : FRAME {} / {}".format(epoch + 1, num_epochs,
                                                     frame, frames[-1]))
        frame_occupants = outlay_dict[frame].keys()
        frame_ptr = frame / __FRAME_GAP
        occu_dict[frame_ptr] = frame_occupants
        hidden_states[frame_ptr] = {}

        for occupant in frame_occupants:
            c = class_dict[occupant]
            position = outlay_dict[frame][occupant]
            model = models[c]

            # images drawing
            if occupant not in real_x:
                real_x[occupant] = []
                real_y[occupant] = []
                predict_x[occupant] = []
                predict_y[occupant] = []

            # get previous hidden(temporal hidden)
            if frame_ptr > 0 and occupant in hidden_states[frame_ptr - 1]:
                prev_hidden = hidden_states[frame_ptr - 1][occupant]
            else:
                # hidden_states[frame_ptr-1][occupant] = []
                prev_hidden = torch.zeros(__HIDDEN_DIM)
                if torch.cuda.is_available():
                    prev_hidden = prev_hidden.cuda()

            # get the spatial hidden
            nei_hidden = []
            for occu in frame_occupants:
                if frame_ptr > 0 and occu in hidden_states[
                        frame_ptr - 1] and occu != occupant:
                    nei_hidden.append(hidden_states[frame_ptr - 1][occu].view(
                        1, __HIDDEN_DIM))

            if len(nei_hidden) > 0:
                spat_hidden = torch.cat(nei_hidden)
            else:
                spat_hidden = torch.zeros(1, __HIDDEN_DIM)

            if torch.cuda.is_available():
                spat_hidden = spat_hidden.cuda()
            # get hiddens
            predict, hidden = model(position, prev_hidden, spat_hidden,
                                    prev_hidden)

            hidden.detach_()

            # used to compute concatenate hiddens
            hidden_states[frame_ptr][occupant] = hidden

            path = path_dict[frame][occupant]
            if len(path) > __OBSERVE_LEN + 1:
                real_xy = path[-1]
                observe_xy = path[-__OBSERVE_LEN - 1:-1]

                prev_hidden = hidden_states[frame_ptr - __OBSERVE_LEN -
                                            1][occupant]

                for i in range(__OBSERVE_LEN):
                    # get the hidden of neighbors at current frame
                    van_hid = []
                    for occu in occu_dict[frame_ptr - __OBSERVE_LEN + i]:
                        if occu != occupant:
                            van_hid.append(
                                hidden_states[frame_ptr - __OBSERVE_LEN +
                                              i][occu].view(1, __HIDDEN_DIM))

                    van_spat_hid = torch.cat(van_hid)
                    if torch.cuda.is_available():
                        van_spat_hid = van_spat_hid.cuda()

                    van_temp_hid = hidden_states[frame_ptr - __OBSERVE_LEN +
                                                 i][occupant]

                    predict, hidden = model(observe_xy[i], van_temp_hid,
                                            van_spat_hid, prev_hidden)
                    prev_hidden = hidden

                loss = loss_fn(predict, real_xy)
                if calculate_loss:
                    cost[c].append(loss.item())
                optimizer = optimizers[c]
                optimizer.zero_grad()
                loss.backward()
                if __GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(models[c].parameters(),
                                                   __GRAD_CLIP)
                optimizer.step()
                # del hidden_states[occupant][-__OBSERVE_LEN-2]
                # image drawing
                if (epoch + 1) % 3 == 0:
                    real_x[occupant].append(real_xy[0].item())
                    real_y[occupant].append(real_xy[1].item())
                    predict_x[occupant].append(predict[0].item())
                    predict_y[occupant].append(predict[1].item())

    # image drawing
    occupants = real_x.keys()
    occupants = sorted(occupants)

    if (epoch + 1) % 3 == 0:
        for occupant in occupants:
            c = class_dict[occupant]
            if len(real_x[occupant]) > 0:
                plt.clf()
                plt.plot(real_x[occupant],
                         real_y[occupant],
                         color='green',
                         linewidth=2,
                         alpha=0.5)
                plt.savefig(
                    'images/epoch{}_scene{}_occupant{}_{}_{}_real.jpg'.format(
                        epoch + 1, s + 1, occupant, len(real_x[occupant]),
                        c.lower()))
                plt.clf()
                plt.plot(predict_x[occupant],
                         predict_y[occupant],
                         color='blue',
                         linewidth=2,
                         alpha=0.5)
                plt.savefig(
                    'images/epoch{}_scene{}_occupant{}_{}_{}_predict.jpg'.
                    format(epoch + 1, s + 1, occupant, len(real_x[occupant]),
                           c.lower()))

    if calculate_loss:
        result = {c: -10000 for c in classes}
        for c in classes:
            if len(cost[c]) > 0:
                result[c] = sum(cost[c]) / len(cost[c])
        return result


def train_with_attn(models,
                    num_scenes,
                    learning_rates,
                    num_epochs,
                    evaluate_loss_after=1):
    prev_cost = {c: float("inf") for c in classes}
    training_set = TrainDataset()
    training_generator = DataLoader(dataset=training_set,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=collate_fn)
    epochs = {s: {c: [] for c in classes} for s in range(4)}
    costs = {s: {c: [] for c in classes} for s in range(4)}
    for epoch in range(num_epochs):
        cost = {c: 0 for c in classes}
        s = 0
        for i in training_generator:
            scene = i[0][0]
            if (epoch + 1) % evaluate_loss_after == 0:
                cost = step_through_scene(models, scene, epoch, num_epochs, s,
                                          True)
                torch.cuda.empty_cache()
                for c in classes:
                    epochs[s][c].append(epoch)
                    costs[s][c].append(cost[c])
                    plt.clf()
                    plt.plot(epochs[s][c],
                             costs[s][c],
                             color='red',
                             linewidth=2,
                             alpha=0.5)
                    plt.savefig('costs/epoch{}_scene{}_{}.jpg'.format(
                        epoch + 1, s + 1, c.lower()))

                    f = open('log.txt', 'a')
                    f.write('epoch{}_scene{}_{}\'cost:{}\n'.format(
                        epoch + 1, s + 1, c.lower(), cost[c]))
                    f.close()

                if (s + 1) == num_scenes:
                    for c in cost:
                        print("{} COST : {}".format(c, cost[c]))
                        if cost[c] > prev_cost[c]:
                            learning_rates[c] *= 0.5
                            print("LEARNING RATE FOR {} WAS HALVED".format(c))
                    prev_cost = cost
                s = s + 1
                continue
            step_through_scene(models, scene, epoch, num_epochs, s, False)
            torch.cuda.empty_cache()
            s = s + 1
        for c in models:
            torch.save(models[c],
                       'models/' + c.lower() + '_' + str(epoch + 1) + '.pkl')


print('creating models')
models = {
    label: AttnGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM, __ATTN_DIM)
    for label in classes
}
if torch.cuda.is_available():
    for c in models:
        models[c] = models[c].cuda()

learning_rates = {c: __LEARNING_RATE for c in classes}
optimizers = {
    c: torch.optim.RMSprop(models[c].parameters(),
                           learning_rates[c],
                           alpha=0.9)
    for c in classes
}
train_with_attn(models, __NUM_SCENES, learning_rates, __NUM_EPOCHS)
