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

def select_hidden_unit(args):
    if args.data_dir == "pamap":
        args.hidden_units = 1000
    elif args.data_dir == "dsads":
        args.hidden_units = 1000
    elif args.data_dir == "housea":
        args.hidden_units = 200
    else:
        args.hidden_units = 500

    return  args.hidden_units

def run_model(identity, method, args, config, train_datasets, test_datasets, verbose=False):
    try:   
        result_folder = args.results_dir

        m, cmd = method

        results = []
        args.replay = m
        identity["method"] = m

        # Use cuda?
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")

        # Set random seeds
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if cuda:
            torch.cuda.manual_seed(args.seed)

        if m == "lwf":
            args.solver_distill = True
        elif m== "ewc":
            args.solver_ewc = True


        identity["cmd"] = str(cmd)

        all_data = None
        for task, train_dataset in enumerate(train_datasets, 1):
            identity["train_session"] = task
            if all_data is None:
                all_data = train_dataset
            else:
                all_data = all_data.merge(train_dataset)

            model = GenerativeReplayLearner(args, 2, verbose=verbose)

            solver = Classifier(
                input_feat=config['feature'],
                classes=len(all_data.classes),
                fc_layers=args.solver_fc_layers, fc_units=args.solver_fc_units, 
                cuda=cuda,
                device=device,
            ).to(device)

            model.set_solver(solver)
            for c in all_data.classes:
                model.classmap.map(c)

            model.train_solver(task, all_data, None)
            result = model.test(task, test_datasets, verbose=verbose)
            results.append(result_to_list(identity, result))

        save_results(result_folder, identity, results)

    except Exception as e:
        print("ERROR:", e)

    print("DONE task order", identity["task_order"])

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
        ("offline", 0),
    ]

    start = time.time()
    ntask = 10

    tasks = []
    if args.task_order is not None:
        ft = open(args.task_order)
        tasks = [line.strip().split(";") for line in ft]

    base_args = args
    for task_order in range(ntask):
        if args.task_order is not None:
            base_dataset.permu_task_order(tasks[task_order])
        else:
            base_dataset.permu_task_order()

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
        if args.task_order is None:
            save_order(result_folder, task_order, base_dataset.classes)

        
        traindata, testdata = base_dataset.train_test_split()

        dataset = traindata
        if args.oversampling:
            dataset = traindata.resampling()

        train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
        test_datasets, _, _ = testdata.split(tasks=args.tasks)
        
        print("CHECK")
        for t in test_datasets:
            print(t.pddata["ActivityName"].unique())

        print("******* Run ",task_order,"*******")
        print("\n")

        for method in methods:
            m, cmd = method
            identity["method"] = m
            args = copy.deepcopy(args)
            
            args.critic_fc_units = select_hidden_unit(args)
            args.generator_fc_units = select_hidden_unit(args)

            run_model(identity, method, args, config, train_datasets, test_datasets, True)
            # pool.apply_async(run_model, args=(identity, method, args, config, train_datasets, test_datasets, True))
            


    training_time = time.time() - start
    print(training_time)

    # clearup_tmp_file(result_folder, ntask, methods)