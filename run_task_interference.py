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
    if args.data_dir == "pamap":
        h = 1000
        args.hidden_units = h
        
    elif args.data_dir == "dsads":
        h = 1000
        args.hidden_units = h

    elif args.data_dir == "housea":
        h = 200
        args.hidden_units = h
    else:
        h = 500
        args.hidden_units = h

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

    methods = [("offline", 0)]

    jobs = []
    # pool = mp.Pool()
    start = time.time()
    ntask = 10

    tasks = [
        [
            "R2_prepare_dinner",
            "R2_watch_TV",
            "R2_prepare_lunch",
            "R1_work_at_dining_room_table",
        ],

        [
            "R2_prepare_dinner",
            "R2_watch_TV",
        ],

        [
            "R2_prepare_lunch",
            "R1_work_at_dining_room_table",
        ],

        [
            "R2_prepare_dinner",
            "R2_prepare_lunch"
        ],

         [
            "R2_watch_TV",
            "R1_work_at_dining_room_table",
        ],
    ]
    
    if args.task_order is not None:
        ft = open(args.task_order)
        tasks = [line.strip().split(";") for line in ft]

    base_args = args
    for task_order in range(len(tasks)):

        base_dataset.permu_task_order(tasks[task_order])
        args.tasks = len(tasks[task_order])//2

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

        # for t in train_datasets:
        #     print(t.classes, args.tasks, taskoooo[task_order])

        for method in methods:
            m, cmd = method
            identity["method"] = m
            args = copy.deepcopy(base_args)
            
            args.critic_fc_units = select_hidden_unit(args, cmd)
            args.generator_fc_units = select_hidden_unit(args, cmd)

            args.g_iters = get_g_iter(m, None)
            model = run_model(identity, method, args, config, train_datasets, test_datasets, True)
            result = model.test(args.tasks, test_datasets, verbose=True)

    training_time = time.time() - start
    print(training_time)

    # clearup_tmp_file(result_folder, ntask, methods)