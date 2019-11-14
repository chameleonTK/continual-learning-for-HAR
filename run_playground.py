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

def select_hidden_unit(args, cmd):
    nFeat = 0
    if args.data_dir == "housea":
        nFeat = 15
    elif args.data_dir == "casas":
        nFeat = 72
    elif args.data_dir == "pamap":
        nFeat = 243
    elif args.data_dir == "dsads":
        nFeat = 405
    else:
        raise Exception("Unknown Dataset")
    
    step = nFeat//4
    args.hidden_units = step * (cmd+1)
    return  args.hidden_units

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
        ("mp-gan", 7),
    ]

    jobs = []
    # pool = mp.Pool()
    start = time.time()

    tasks = []
    if args.task_order is not None:
        ft = open(args.task_order)
        tasks = [line.strip().split(";") for line in ft]
    else:
        tasks = [base_dataset.classes]

    base_args = args
    for task_order in range(len(tasks)):
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
        
        
        identity["task_order"] = task_order

        
        traindata, testdata = base_dataset.train_test_split()

        dataset = traindata
        if args.oversampling:
            dataset = traindata.resampling()

        train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
        test_datasets, _, _ = testdata.split(tasks=args.tasks)

        print("******* Run ",task_order,"*******")
        print("\n")

        for method in methods:
            m, cmd = method
            identity["method"] = m
            args = copy.deepcopy(base_args)
        
            print("#Neuron", select_hidden_unit(args, cmd))
            print("#Layer", args.solver_fc_layers-2)

            args.critic_fc_layers = args.solver_fc_layers
            args.generator_fc_layers = args.solver_fc_layers
            args.solver_fc_units = select_hidden_unit(args, cmd)
            args.critic_fc_units = select_hidden_unit(args, cmd)
            args.generator_fc_units = select_hidden_unit(args, cmd)

            args.g_iters = get_g_iter(m, None)

            print(args.critic_fc_layers)
            run_model(identity, method, args, config, train_datasets, test_datasets, True)
            

    training_time = time.time() - start
    print(training_time)