import json
import sys

import numpy as np
from PIL import Image

TRAIN_SEQS = [
    1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48,
    50, 52, 53, 54
]
VAL_SEQS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TRAINVAL_SEQS = [
    1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48,
    50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51
]
TEST_SEQS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

ACTIONS = [
    'blocking', 'digging', 'falling', 'jumping', 'moving', 'setting',
    'spiking', 'standing', 'waiting'
]
NUM_ACTIONS = 9

ACTIVITIES = [
    'r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass',
    'l_winpoint'
]
NUM_ACTIVITIES = 8
HIGH_RESOLUTION = [2, 37, 38, 39, 40, 41, 44, 45]

FRAME_NUM = 8

anno_name = 'tracklets.txt'

gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
act_to_id = {name: i for i, name in enumerate(ACTIONS)}

TRAINVAL_PIXEL_PATH = "./data/trainval_bboxes_{}frames.tsv".format(FRAME_NUM)
TEST_PIXEL_PATH = "./data/test_bboxes_{}frames.tsv".format(FRAME_NUM)

TARGET_SIZE = (1280, 720)
TARGET_FRAME = 21


def volley_read_annotations(img_dir, track_path, sid, phase):
    if sid in HIGH_RESOLUTION:
        H = 1080.0
        W = 1920.0
    else:
        H = 720.0
        W = 1280.0
    anno_dir = track_path + str(sid) + '/'
    activity_path = img_dir + '%d/annotations.txt' % sid

    if phase == 'train':
        tsv_path = TRAIN_PIXEL_PATH
    elif phase == 'val':
        tsv_path = VAL_PIXEL_PATH
    elif phase == 'trainval':
        tsv_path = TRAINVAL_PIXEL_PATH
    else:
        tsv_path = TEST_PIXEL_PATH

    print('phase {}'.format(phase))
    print('opening {} ...'.format(activity_path))
    with open(activity_path) as grp_f:
        for l in grp_f.readlines():
            grp_values = l[:-1].split(' ')
            file_name = grp_values[0]
            tf_id = int(file_name.split('.')[0])
            fids = range(tf_id - FRAME_NUM / 2, tf_id + FRAME_NUM / 2)

            activity = gact_to_id[grp_values[1]]

            grp_values = grp_values[2:]
            num_people = len(grp_values) // 5

            person_actions = []
            anno_path = anno_dir + str(tf_id) + '/'
            anno = anno_path + anno_name

            bboxes = {fidx: [] for fidx in range(FRAME_NUM)}
            actions = []
            action_names = []

            # get bboxes
            print('opening {} ...'.format(anno))
            with open(anno) as ind_f:
                for l in ind_f.readlines():
                    ind_values = l[:-1].split('\t')
                    action_name = ind_values[0]
                    action_names.append(action_name)

                    action = act_to_id[action_name]
                    actions.append(action)
                    for fidx, bbox in enumerate(
                            ind_values[TARGET_FRAME -
                                       (FRAME_NUM / 2):TARGET_FRAME +
                                       (FRAME_NUM / 2)]):
                        # print(fidx, bbox)
                        x, y, w, h = map(float, bbox.split(' '))
                        # bboxes[fidx].append((x, y, w, h))
                        bboxes[fidx].append(
                            (y / H, x / W, (y + h) / H, (x + w) / W))

            bboxes_frame = []
            for fidx in bboxes:
                bboxes_frame.append(bboxes[fidx])
            # [T, N]

            for pid in range(num_people):
                person_actions.append(actions[pid])

            json_dict = {
                'video': sid,
                'tf': tf_id,
                'glabel': activity,
                'plabel': person_actions,
                'bboxes': bboxes_frame,
            }
            print('writing {} ...'.format(tsv_path))
            with open(tsv_path, 'a') as tsv_file:
                tsv_file.write('{}\n'.format(json.dumps(json_dict)))


def volley_read_dataset(img_path, trac_path, seqs, phase):
    for sid in seqs:
        volley_read_annotations(img_path, trac_path, sid, phase)
        # break


img_dir = "../dataset/Volleyball/videos/"
trac_path = "../dataset/Volleyball/volleyball-extra/"

volley_read_dataset(img_dir, trac_path, TRAINVAL_SEQS, 'trainval')
volley_read_dataset(img_dir, trac_path, TEST_SEQS, 'test')