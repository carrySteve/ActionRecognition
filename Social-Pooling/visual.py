import sys

import matplotlib.pyplot as plt

sp_path = "sp.txt"
sa_path = "sa.txt"

scenes = ['1', '2', '3', '4']
classes = ['pedestrian', 'biker', 'skater', 'cart']
sp_cost = {s: {c: [] for c in classes} for s in scenes}
sa_cost = {s: {c: [] for c in classes} for s in scenes}

with open(sp_path) as f:
    for l in f.readlines():
        values = l[:-1].split('_')
        epoch = int(values[0][5:])
        scene = values[1][-1]

        occu_cost = values[-1].split(':')
        occu = occu_cost[0].split('\'')[0]
        cost = float(occu_cost[1])

        if epoch == 1:
            continue
        elif epoch > 57:
            break
        else:
            sp_cost[scene][occu].append(cost)
            
with open(sa_path) as f:
    for l in f.readlines():
        values = l[:-1].split('_')
        epoch = int(values[0][5:])
        scene = values[1][-1]

        occu_cost = values[-1].split(':')
        occu = occu_cost[0].split('\'')[0]
        cost = float(occu_cost[1])

        if epoch == 1:
            continue
        elif epoch > 57:
            break
        else:
            sa_cost[scene][occu].append(cost)

num_epochs = 57

for s in scenes:
    for c in classes:
        plt.cla()
        plt.title(
            "Scene {} {} \n Social Pooling Accuracy vs. Social Attention Accuracy"
            .format(s, c))
        plt.xlabel("Training Epochs")
        plt.ylabel("Training Accuracy")
        plt.plot(
            range(2, num_epochs + 1),
            sp_cost[s][c],
            "x-",
            color='green',
            linewidth=2,
            alpha=0.5,
            label="Social Pooling")
        plt.plot(
            range(2, num_epochs + 1),
            sa_cost[s][c],
            "+-",
            color='red',
            linewidth=2,
            alpha=0.5,
            label="Social Attention")
        if c == 'skater' or c == 'biker':
            plt.ylim((0, 3500))
        else:
            plt.ylim((0, 1500))
        plt.legend()
        plt.savefig('scene{}_{}\'s cost'.format(s, c))
