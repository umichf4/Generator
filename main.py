# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
import json
from train_and_test_2 import train_generator, test_generator, train_simulator, test_simulator
from utils import Params
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--save_model_dir', default='models', type=str)
parser.add_argument('--restore_from', default=None, type=str)
parser.add_argument('--json_path', default='params.json', type=str)
parser.add_argument('--T_path', default='data\\shape_spec_3881.mat', type=str)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
args = parser.parse_args()

# Load parameters from json file
json_path = os.path.join(current_dir, args.json_path)
assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
params = Params(json_path)
params.restore_from = args.restore_from
params.save_model_dir = args.save_model_dir
params.T_path = args.T_path
params.freeze = args.freeze
params.cuda = torch.cuda.is_available()

if args.test:
    test_simulator(params)
else:
    train_simulator(params)
