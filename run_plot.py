#!/usr/bin/env python3
import torch
import numpy as np
from smart_home_dataset import SmartHomeDataset
from classifier import Classifier
from torch import optim
import utils
import callbacks as cb
import time

from generative_replay_learner import GenerativeReplayLearner;
import arg_params
import json
import os
import copy

import torch.multiprocessing as mp
from run_main import *

def get_g_iter(method, cmd=None):
    return 1000*int(cmd)

def get_hidden_unit(args):
    if args.data_dir == "pamap":
        return 500

    elif args.data_dir == "dsads":
        return 1000

    elif args.data_dir == "housea":
        return 100

    else:
        return 100

if __name__ == "__main__":

    parser = arg_params.get_parser()
    args = parser.parse_args()
    print("Arguments")
    print(args)

    result_folder = args.results_dir

    print("\n")
    print("STEP1: load datasets")

    base_dataset = select_dataset(args)
    


    methods = [ 
        ("offline", 0), ("none", 0), ("exact", 0), ("mp-gan", 0), ("mp-wgan", 0), ("sg-cgan", 0), ("sg-cwgan", 0), ("lwf", 0), ("ewc", 0),
    ]

    
    
    identity = {
        "task_order": None,
        "method": None,
        "train_session": None,
        "task_index": None,
        "no_of_test": None,
        "no_of_correct_prediction": None,
        "accuracy": None,
        "solver_training_time": None,
        "generator_training_time": None,
    }
    
    
    identity["task_order"] = 1

    
    traindata, testdata = base_dataset.train_test_split()

    dataset = traindata

    if args.oversampling:
        dataset = traindata.resampling()
    
    
    train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
    test_datasets, _, _ = testdata.split(tasks=args.tasks)
    
    base_args = args
    for method in methods:
        start = time.time()

        m, cmd = method
        identity["method"] = m
        args = copy.deepcopy(base_args)
        
        args.critic_fc_units = get_hidden_unit(args)
        args.generator_fc_units = get_hidden_unit(args)

        args.g_iters = get_g_iter(m, cmd+1)

        env_name = "Continual learning ["+m+"]"
        visdom = {'env': env_name, 'graph': "models"}

        run_model(identity, method, args, config, train_datasets, test_datasets, verbose=True, visdom=visdom)
            
        training_time = time.time() - start

        print("Training Time", training_time)

    