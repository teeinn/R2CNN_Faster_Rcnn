# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'FDDB':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'face': 1
    }
elif cfgs.DATASET_NAME == 'ICDAR2015':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'text': 1
    }
elif cfgs.DATASET_NAME.startswith('DOTA'):
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15
    }
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
elif cfgs.DATASET_NAME == 'ROOF':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'flatroof': 1,
        'facility': 2,
        'rooftop': 3,
        "solarpanel_flat": 4,
        "solarpanel_slope": 5,
        "parkinglot" : 6,
        "garage" : 7,
        "smallroof" : 8,
        "small_flat" : 9
    }
# elif cfgs.DATASET_NAME == 'ROOF':
#     NAME_LABEL_MAP = {
#         'back_ground': 0,
#         'flatroof': 1,
#         'facility': 2
#     }
# elif cfgs.DATASET_NAME == 'ROOF':
#     NAME_LABEL_MAP = {
#         'back_ground': 0,
#         'cluster0': 1,
#         'cluster1': 2
#     }
# elif cfgs.DATASET_NAME == 'ROOF':
#     NAME_LABEL_MAP = {
#         'back_ground': 0,
#         'flatroof': 1,
#         'solarpanel_flat': 2,
#         'solarpanel_slope': 3,
#         'parkinglot': 4,
#         'facility': 5,
#         'rooftop': 6,
#         'heliport_r' : 7,
#         'heliport_h' : 8
#     }
# elif cfgs.DATASET_NAME == 'ROOF':
#     NAME_LABEL_MAP = {
#         'back_ground': 0,
#         'goodroof': 1,
#         'solarpanel': 2,
#         'parkinglot': 3,
#         'facility': 4,
#         'rooftop': 5
#     }

else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()
