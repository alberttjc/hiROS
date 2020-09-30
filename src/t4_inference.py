#!/usr/bin/env python
import os
import sys
import argparse
import time
import random
import signal
import numpy as np
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

from loader import TextLoader
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

def main(args):
    dataset_name	=	"Sample"
    model_name		=	"lstm100"
    model_dir		=	os.path.join(args.model_dir, dataset_name)
    ckpt_file		=	os.path.join(model_dir, model_name + ".ckpt")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
		#os.makedirs(os.path.join(model_dir, 'plots'))
    print("=> Output folder for this run -- {}".format(ckpt_file))

    if args.use_gpu:
        gpus = [int(i) for i in args.gpus.split(',')]
        print("=> active GPUs: {}".format(args.gpus))

    model 			= 	LSTMClassifier(args.input_size, args.num_layers, args.hidden_size, args.seq_len, args.num_classes)
    model 	        = 	model.cuda() if use_cuda else model
    model.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))
    dataset_test 	= 	TextLoader(data_dir = args.data_dir, transform = transforms.ToTensor)
    test_loader 	= 	DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True)

    for batch_idx, (data, targets) in enumerate(test_loader):
        #print(data.shape)
        print(data.size())
        """
        data        =   data.to(device=device)
        targets     =   targets.to(device=device)

        scores      =   model(torch.autograd.Variable(data))
        prediction	=	torch.max(scores, 1)[1]



        for e1 in enumerate(prediction):
            print(LABELS[e1[1].item()])
        """
# Constants used - more for a reminder
#	input_size 	= 36
#	num_layers 	= 2
#	hidden_size = 34
#	seq_len		= 32
#	num_classes = 6

str2bool = lambda x: (str(x).lower() == 'true')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', 	type=str, default='checkpoints/',
											help='data_directory')
	parser.add_argument('--data_dir', 		type=str, default='/home/albie/Data/',
											help='data_directory')
	parser.add_argument('--hidden_size',	type=int, default=34,
											help='LSTM hidden dimensions')
	parser.add_argument('--batch_size', 	type=int, default=128,
											help='size for each minibatch')
	parser.add_argument('--input_size', 	type=int, default=36,
											help='x and y dimension for 18 joints')
	parser.add_argument('--num_layers', 	type=int, default=2,
											help='number of hidden layers')
	parser.add_argument('--seq_len', 		type=int, default=30,
											help='number of steps/frames of each action')
	parser.add_argument('--num_classes',	type=int, default=7,
											help='number of classes/type of each action')
	parser.add_argument('--learning_rate',	type=float, default=0.001,
											help='initial learning rate')
	parser.add_argument('--weight_decay', 	type=float, default=1e-5,
											help='weight_decay rate')

	parser.add_argument('--use_gpu',		type=str2bool, default=False,
											help="flag to use gpu or not.")
	parser.add_argument('--gpus',			type=int, default=0,
											help='gpu ids for use')
	parser.add_argument('--transfer',		type=str2bool, default=False,
											help='resume training from given checkpoint')

	args = parser.parse_args()
	main(args)
