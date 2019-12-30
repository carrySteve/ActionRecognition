#training script for neural nets
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_dataset import TrainDataset
from models import PoolingGRU
from torch.autograd import Variable

import argparse
import math
import matplotlib.pyplot as plt

_id = 0
_position = 1
_class = 2

#hyperparameters
__INPUT_DIM = 2
__OUTPUT_DIM = 6
__HIDDEN_DIM = 128
__NUM_EPOCHS = 100
__LEARNING_RATE = 0.003
__POOLING_SIZE = 20
__NUM_SCENES = 4
__GRAD_CLIP = 0.25
__FRAME_GAP = 15
__GAUSSIAN_NUM = 1
__IS_TRAINING = True

classes = ["Pedestrian", "Biker", "Skater", "Cart"]

loss_fn = torch.nn.CrossEntropyLoss()  # !! size_average, ignore_index, reduce, reduction


def collate_fn(data):
    x, y = zip(*data)
    return x, y


def gaussian_pdf(x, y, pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy, gaussian_num):
    z_x = ((x-u_x)/sigma_x)**2
    z_y = ((y-u_y)/sigma_y)**2
    z_xy = (x-u_x)*(y-u_y)/(sigma_x*sigma_y)
    z = z_x + z_y - 2*rho_xy*z_xy
    a = -z/(2*(1-rho_xy**2))

    a = a.view(gaussian_num)
    a_max = torch.max(a, dim=0)[0]
    a_max = a_max.unsqueeze(0).repeat(1, gaussian_num)
    print(a_max)
    a, a_max = a.view(-1), a_max.view(-1)
    print(a_max)

    exp = torch.exp(a-a_max)
    norm = torch.clamp(2*np.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2), min=1e-5)
    raw_pdf = pi_xy*exp/norm
    raw_pdf = raw_pdf.view(gaussian_num)
    raw_pdf = torch.log(torch.sum(raw_pdf, dim=0)+1e-5)
    a_max = a_max.view(gaussian_num)[:,0]
    raw_pdf = raw_pdf+a_max

    return raw_pdf


def predict_fn(model, XYs, H, real_xy):
    prev_hidden = torch.zeros(__HIDDEN_DIM, )
    if torch.cuda.is_available():
        prev_hidden = prev_hidden.cuda()

    for i in range(18):
        predict, hidden = model(XYs[i], H[i], prev_hidden)
        prev_hidden = hidden

    return predict


def map_tensor_index(other_position, ref_pos):
    x = math.ceil((other_position[0] - ref_pos[0])/8) + 9
    y = math.ceil((other_position[1] - ref_pos[1])/8) + 9
    return (int(x), int(y))


def pool_hidden_states(member_id, position, hidden_states):
    pooled_tensor = torch.zeros(__POOLING_SIZE, __POOLING_SIZE, __HIDDEN_DIM)

    if torch.cuda.is_available():
        pooled_tensor = pooled_tensor.cuda()

    bound = __POOLING_SIZE * 8 / 2
    window_limits_upper_bound = (position[0] + bound, position[1] + bound)
    window_limits_lower_bound = (position[0] - bound, position[1] - bound)
    for ID in hidden_states:  # try all hidden_states
        if ID != member_id:
            other_position = hidden_states[ID][0]
            within_upper_bound = (other_position[0] <= window_limits_upper_bound[0]) and (other_position[1] <= window_limits_upper_bound[1])
            within_lower_bound = (other_position[0] > window_limits_lower_bound[0]) and (other_position[1] > window_limits_lower_bound[1])
            if within_upper_bound and within_lower_bound:
                x, y = map_tensor_index(other_position, position)
                pooled_tensor[x][y] = pooled_tensor[x][y] + hidden_states[ID][1]
    return pooled_tensor


def step_through_scene(models, scene, epoch, num_epochs, s, calculate_loss, is_training):
    outlay_dict = scene[0]
    class_dict = scene[1]
    path_dict = scene[2]
    frames = outlay_dict.keys()
    frames = sorted(frames)
    prev_hiddens = {}
    pooled_tensors = {}

    if not is_training:
        real_x = {}
        real_y = {}
        predict_x = {}
        predict_y = {}

    cost = {c: [] for c in classes}

    for frame in frames:
        print "EPOCH {} / {} : FRAME {} / {}".format(epoch+1, num_epochs, frame, frames[-1])
        frame_occupants = outlay_dict[frame].keys()
        hidden_states = {}
        for occupant in frame_occupants:
            c = class_dict[occupant]
            position = outlay_dict[frame][occupant]

            if not is_training:
                if occupant not in real_x:
                    real_x[occupant] = []
                    real_y[occupant] = []
                    predict_x[occupant] = []
                    predict_y[occupant] = []

            # social pooling
            if occupant not in pooled_tensors:
                pooled_tensors[occupant] = []
            pooled_hidden = pool_hidden_states(occupant, position.cpu().tolist(), hidden_states)
            pooled_tensors[occupant].append(pooled_hidden)

            model = models[c]
            if occupant in prev_hiddens:
                prev_hidden = prev_hiddens[occupant]
            else:
                prev_hidden = torch.zeros(__HIDDEN_DIM)
                if torch.cuda.is_available():
                    prev_hidden = prev_hidden.cuda()
            predict, hidden = model(position, pooled_hidden, prev_hidden)

            hidden.detach_()
            prev_hiddens[occupant] = hidden
            hidden_states[occupant] = (position.cpu().tolist(), hidden)

            path = path_dict[frame][occupant]
            if len(path) > 18:
                real_xy = path[-1]
                XYs = path[-19:-1]
                H = pooled_tensors[occupant][-18:]
                predict = predict_fn(model, XYs, H, real_xy)  # gaussian params when training, coordinates when testing

                if is_training:
                    pi, u_x, u_y, sigma_x, sigma_y, rho_xy = predict  # (1,) (1,) (1,) (1,) (1,)
                    XYs = path[-18:]
                    log_sum = torch.Tensor([0])
                    for i in range(18):
                        real_x = XYs[i][0]
                        real_y = XYs[i][1]
                        pdf = gaussian_pdf(real_x, real_y, pi, u_x, u_y, sigma_x, sigma_y, rho_xy, __GAUSSIAN_NUM)
                        # log? soft-max? loge?
                        loss = -torch.log(log_sum+pdf)

                    if calculate_loss:
                        cost[c].append(loss.item())
                    optimizer = optimizers[c]
                    optimizer.zero_grad()
                    loss.backward()
                    if __GRAD_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(models[c].parameters(), __GRAD_CLIP)
                    optimizer.step()
                    del pooled_tensors[occupant][-18]

                # image drawing
                if not is_training:
                    if (epoch + 1) % 3 == 0:
                        real_x[occupant].append(real_xy[0].item())
                        real_y[occupant].append(real_xy[1].item())
                        predict_x[occupant].append(predict[0].item())
                        predict_y[occupant].append(predict[1].item())

            if frame+__FRAME_GAP <= frames[-1] and occupant not in outlay_dict[frame+__FRAME_GAP].keys():
                for h in pooled_tensors[occupant]:
                    del h

    if not is_training:
        # image drawing
        occupants = real_x.keys()
        occupants = sorted(occupants)
        if (epoch + 1) % 3 == 0:
            for occupant in occupants:
                c = class_dict[occupant]
                if len(real_x[occupant]) > 0:
                    plt.clf()
                    plt.plot(real_x[occupant], real_y[occupant], color='green',linewidth=2, alpha=0.5)
                    plt.savefig('images/epoch'+ str(epoch+1) + '_scene' + str(s+1) + 
                        '_occupant'+ str(occupant) + '_' + str(len(real_x[occupant]))+ '_' + c.lower() +'_real.jpg')
                    plt.clf()
                    plt.plot(predict_x[occupant], predict_y[occupant], color='blue',linewidth=2, alpha=0.5)
                    plt.savefig('images/epoch'+ str(epoch+1) + '_scene' + str(s+1) +
                    '_occupant'+ str(occupant) + '_' + str(len(predict_x[occupant]))+ '_' + c.lower() + '_predict.jpg')

    if calculate_loss:
        result = {c: -10000 for c in classes}
        for c in classes:
            if len(cost[c]) > 0:
                result[c] = sum(cost[c])/len(cost[c])
        return result


def train_with_pooling(models, num_scenes, learning_rates, num_epochs, evaluate_loss_after=1, is_training=True):
    prev_cost = {c: float("inf") for c in classes}
    training_set = TrainDataset()
    training_generator = DataLoader(dataset=training_set, batch_size=1, 
        shuffle=False, collate_fn=collate_fn)
    epochs = {s: {c: [] for c in classes} for s in range(4)}
    costs = {s: {c: [] for c in classes} for s in range(4)}
    for epoch in range(num_epochs):
        cost = {c: 0 for c in classes}
        s = 0
        for i in training_generator:
            scene = i[0][0]
            if (epoch + 1) % evaluate_loss_after == 0:
                cost = step_through_scene(models, scene, epoch, num_epochs, s, True, __IS_TRAINING)
                torch.cuda.empty_cache()
                for c in classes:
                    epochs[s][c].append(epoch)
                    costs[s][c].append(cost[c])
                    plt.clf()
                    plt.plot(epochs[s][c], costs[s][c], color='red',linewidth=2, alpha=0.5)
                    plt.savefig('costs/' + 'epoch' + str(epoch+1) + '_scene' + str(s+1) + '_' + c.lower() + '.jpg')

                if (s+1) == num_scenes:
                    for c in cost:
                        print "{} COST : {}".format(c, cost[c])
                        if cost[c] > prev_cost[c]:
                            learning_rates[c] *= 0.5
                            print "LEARNING RATE FOR {} WAS HALVED".format(c)
                    prev_cost = cost
                s = s + 1
                continue
            step_through_scene(models, scene, epoch, num_epochs, s, False, __IS_TRAINING)
            torch.cuda.empty_cache()            s = s + 1
        for c in models:
            torch.save(models[c], 'models/' + c.lower() + '_' + str(epoch+1) + '.pkl')


parser = argparse.ArgumentParser(description='Pick Training Mode.')
parser.add_argument('mode', type=str, nargs=1, help="which mode to use for training? either 'pooling' or 'naive'")
mode = parser.parse_args().mode[0]

if mode == "pooling":
    print 'creating models'
    models = {label: PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __POOLING_SIZE, __GAUSSIAN_NUM, __HIDDEN_DIM, __IS_TRAINING) for label in classes}
    if torch.cuda.is_available():
        for c in models:
            models[c] = models[c].cuda()

    learning_rates = {c: __LEARNING_RATE for c in classes}
    optimizers = {c: torch.optim.RMSprop(models[c].parameters(), learning_rates[c], alpha=0.9) for c in classes}
    train_with_pooling(models, __NUM_SCENES, learning_rates, __NUM_EPOCHS)

else:
    print("enter a valid mode: either 'pooling' or 'naive'")
