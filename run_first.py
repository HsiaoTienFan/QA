# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:40:39 2018

@author: fanat
"""

import preprocess
from Models import model_rnet
import numpy as np
import tensorflow as tf
import argparse
import random
import json
from pprint import pprint
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rnet', help='Model: match_lstm, bidaf, rnet')
parser.add_argument('--debug', type=bool, default=False, help='print debug msgs')
parser.add_argument('--dataset', type=str, default='dev', help='dataset')
parser.add_argument('--model_path', type=str, default='Models/save/rnet_model27.ckpt', help='saved model path')

args = parser.parse_args()
if not args.model == 'rnet':
	raise NotImplementedError

modOpts = json.load(open('Models/config.json','r'))[args.model]['dev']
print('Model Configs:')
pprint(modOpts)

print('Reading data')
if args.dataset == 'train':
	raise NotImplementedError
elif args.dataset == 'dev':
	dp = preprocess.read_data(args.dataset, modOpts)
    
model = model_rnet.R_NET(modOpts)
input_tensors, loss, acc, pred_si, pred_ei = model.build_model()
saved_model = args.model_path


num_batches = int(np.ceil(dp.num_samples/modOpts['batch_size']))
print(num_batches, 'batches')
	
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
new_saver = tf.train.Saver()
sess = tf.InteractiveSession(config=config)
new_saver.restore(sess, saved_model)
	
pred_data = {}

EM = 0.0
F1 = 0.0
empty_answer_idx = np.ndarray((modOpts['batch_size'], modOpts['p_length']))