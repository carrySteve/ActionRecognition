import torch
from torch.utils.data import Dataset
import pickle
import sys
import os


class TrainDataset(Dataset):
    def __init__(self):
        self.NUM_SCENES = 5
        self.TRAIN_SCENE = 4
        for s in range(self.NUM_SCENES):
            #load annotations

            dataset = open('annotations/deathCircle/video' + str(s) +
                           '/annotations.txt')
            #dictionary to hold parsed details
            scene = {}

            while True:
                line = dataset.readline()
                if line == '':
                    break
                row = line.split(" ")
                frame = int(row[5])
                if frame % 15 != 0:
                    continue

                x = (float(row[1]) + float(row[3])) / 2
                y = (float(row[2]) + float(row[4])) / 2
                label = row[-1][1:-2]
                #skip sparse busses and resolve cars as carts
                if label == "Bus":
                    continue
                if label == "Car":
                    label = "Cart"
                member_id = int(row[0])

                position = (x, y)
                position = torch.Tensor(position)

                if torch.cuda.is_available():
                    position = position.cuda()

                info = [member_id, position, label]
                if frame in scene:
                    scene[frame].append(info)
                else:
                    scene[frame] = [info]

            #spearate parsed info into the three dictionaries (reduces complexity while training)
            #outlay_dict contains position per frame. class_dict contains classification per member-id. path_dict contains path thus far per member-id
            outlay_dict, class_dict, path_dict = {}, {}, {}
            frames = scene.keys()
            frames = sorted(frames)
            for frame in frames:
                outlay_dict[frame], path_dict[frame] = {}, {}
                for obj in scene[frame]:
                    outlay_dict[frame][obj[0]] = obj[1]
                    class_dict[obj[0]] = obj[2]

                    if frame == 0:
                        path_dict[frame][obj[0]] = [obj[1]]
                        continue

                    prev_frame = frames[frames.index(frame) - 1]
                    if obj[0] not in path_dict[prev_frame]:
                        path_dict[frame][obj[0]] = [obj[1]]
                    else:
                        path_dict[frame][
                            obj[0]] = path_dict[prev_frame][obj[0]] + [obj[1]]

            if os.path.exists('train_data/pooling/scene' + str(s) +
                              '/scene.pickle'):
                pass
            else:
                pickle.dump(
                    [outlay_dict, class_dict, path_dict],
                    open('train_data/pooling/scene' + str(s) + '/scene.pickle',
                         'wb'))

    def __getitem__(self, index):
        y = 'train_data/pooling/scene' + str(index) + '/scene.pickle'
        X = pickle.load(open(y, 'rb'))
        return X, y

    def __len__(self):
        return self.TRAIN_SCENE