#!/usr/bin/env python
import os
import sys
import argparse
import time
import random
import signal
import numpy as np
import cv2
#import matplotlib
#import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
#import torch.functional as F
#import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from tqdm import tqdm
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from collections import OrderedDict, deque

# Pose Estimation
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


from model import LSTMClassifier

LABELS = [
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "WAVE",
    "CLAP",
    "STAND",
    "CLOCKWISE",
    "COUNTER_CLOCKWISE",
]


use_cuda    =   torch.cuda.is_available()
device	    =	torch.device("cuda" if use_cuda else "cpu")

str2bool = lambda x: (str(x).lower() == 'true')

def humans_to_skels_list(humans, scale_h):
    ''' Get skeleton data of (x, y * scale_h) from humans.
    Arguments:
        humans {a class returned by self.detect}
        scale_h {float}: scale each skeleton's y coordinate (height) value.
            Default: (image_height / image_widht).
    Returns:
        skeletons {list of list}: a list of skeleton.
            Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
        scale_h {float}: The resultant height(y coordinate) range.
            The x coordinate is between [0, 1].
            The y coordinate is between [0, scale_h]
    '''
    skeletons = []
    NaN = 0
    for human in humans:
        skeleton = [NaN]*(18*2)
        for i, body_part in human.body_parts.items(): # iterate dict
            idx = body_part.part_idx
            skeleton[2*idx]=body_part.x
            skeleton[2*idx+1]=body_part.y * scale_h
        skeletons.append(skeleton)
    return skeletons, scale_h

def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons

def main(args):
    # Constants
    Q               =   deque(maxlen=30)
    SQ              =   deque(maxlen=10)
    act             =   deque(['No gesture','No gesture'], maxlen=3)
    # Webcam initialization
    w, h            =   model_wh(args.resize)
    e               =   TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(False))
    cam             =   cv2.VideoCapture(0)
    ret_val, image  =   cam.read()
    _scale_h        =   1.0 * image.shape[0] / image.shape[1]

    # Model initialization
    print("=> Output folder for this run -- {}".format(args.ckpt_file))

    if args.use_gpu:
        gpus = [int(i) for i in args.gpus.split(',')]
        print("=> active GPUs: {}".format(args.gpus))

    model   =   LSTMClassifier(args.input_size, args.num_layers, args.hidden_size, args.seq_len, args.num_classes)
    model   = 	model.cuda() if use_cuda else model

    model.load_state_dict(torch.load(args.ckpt_file, map_location=torch.device('cpu')))

    while True:
        ret_val, image      =   cam.read()

        # Detect skeletons
        humans              =   e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        skeletons, scale_h  =   humans_to_skels_list(humans,_scale_h)
        skeletons           =   remove_skeletons_with_few_joints(skeletons)

        # -- Track people
        #dict_id2skeleton = multiperson_tracker.track(skeletons)

        Q.append(skeletons)
        frames  = []
        for frame_idx in Q:
            frame_idx = torch.tensor(frame_idx)
            frames.append(frame_idx)

        data = torch.cat(frames)
        data = data.to(device=device)
        data = data[None,:]

        if data.size()[1] is 30:
            output = model(data)
            output = torch.nn.functional.softmax(output, dim=1)

            k = 7
            threshold = 0.7
            ts, pred = output.detach().cpu().topk(k, 1, True, True)
            top5 = [LABELS[pred[0][i].item()] for i in range(k)]

            pi = [pred[0][i].item() for i in range(k)]
            ps = [ts[0][i].item() for i in range(k)]
            top1 = top5[0] if ps[0] > threshold else LABELS[0]

            hist = {}

            for i in range(7):
                hist[i] = 0

            for i in range(len(pi)):
                hist[pi[i]] = ps[i]
            #scores      =   model(torch.autograd.Variable(data))
            #prediction	=	torch.max(scores, 1)[1]

            SQ.append(list(hist.values()))
            ave_pred = np.array(SQ).mean(axis=0)
            top1 = LABELS[np.argmax(ave_pred)] if max(ave_pred) > threshold else LABELS[0]

            top1 = top1.lower()
            act.append(top1)

            print(act)
            #print(LABELS[prediction.item()])

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

# Constants used - more for a reminder
#	input_size 	= 36
#	num_layers 	= 2
#	hidden_size = 34
#	seq_len		= 32
#	num_classes = 6

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Webcam configuratrions
    parser.add_argument('--model',              type=str,   default='cmu',   help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize',             type=str,   default='0x0',              help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio',   type=float, default=4.0,                help='if provided, resize heatmaps before they are post-processed. default=1.0')

    # Model configuratrions
    parser.add_argument('--ckpt_file', 	        type=str,   default='checkpoints/Sample/lstm100.ckpt',     help='data_directory')
    parser.add_argument('--hidden_size',        type=int,   default=34,                 help='LSTM hidden dimensions')
    parser.add_argument('--input_size', 	    type=int,   default=36,                 help='x and y dimension for 18 joints')
    parser.add_argument('--num_layers', 	    type=int,   default=2,                  help='number of hidden layers')
    parser.add_argument('--seq_len', 		    type=int,   default=30,                 help='number of steps/frames of each action')
    parser.add_argument('--num_classes',	    type=int,   default=7,                  help='number of classes/type of each action')
    parser.add_argument('--use_gpu',		    type=str2bool, default=False,           help="flag to use gpu or not.")
    parser.add_argument('--gpus',			    type=int,   default=0,                  help='gpu ids for use')

    args = parser.parse_args()
    main(args)
