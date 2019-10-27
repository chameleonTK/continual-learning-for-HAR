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

from scipy import stats

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
    jobs = []
    # pool = mp.Pool()
    start = time.time()

    base_args = args
    
    traindata, testdata = base_dataset.train_test_split()

    dataset = traindata
    if args.oversampling:
        dataset = traindata.resampling()

    train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
    test_datasets, _, _ = testdata.split(tasks=args.tasks)
    test_datasets_per_class, _, _ = testdata.split(tasks=len(testdata.classes))

    results = []
    for i in range(10):
        for method in methods:
            m, cmd = method
            args = copy.deepcopy(base_args)

            args.seed = i
            args.critic_fc_units = select_hidden_unit(args, cmd)
            args.generator_fc_units = select_hidden_unit(args, cmd)

            args.g_iters = get_g_iter(m, None)
            model = run_model(identity, method, args, config, train_datasets, test_datasets, True)

            print("\nManual test")
            for t in test_datasets_per_class:
                print(t, len(t), t.pddata["ActivityName"].unique())

            result = model.test(None, test_datasets_per_class, verbose=True)
            results.append(result)
    
    
    print("Offline Accuracy")
    for i in range(len(testdata.classes)):
        c = testdata.classes[i]
        val = []
        for r in results:
            val.append(r["Precision"][i])

        avg = np.mean(val)
        err = stats.sem(val)
        print("{0} {1:.3f} {2:.3f}".format(c, avg, err))


    training_time = time.time() - start
    print(training_time)

    # clearup_tmp_file(result_folder, ntask, methods)