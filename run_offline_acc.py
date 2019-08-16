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

        model = GenerativeReplayLearner(args, 2, verbose=verbose)

        solver = Classifier(
            input_feat=config['feature'],
            classes=len(train_datasets[0].classes),
            fc_layers=args.solver_fc_layers, fc_units=args.solver_fc_units, 
            cuda=cuda,
            device=device,
        ).to(device)
        model.set_solver(solver)

        all_data = None
        for task, train_dataset in enumerate(train_datasets, 1):
            for c in train_dataset.classes:
                model.classmap.map(c)

            if all_data is None:
                all_data = train_dataset
            else:
                all_data = all_data.merge(train_dataset)

            if task==1:
                continue
            
            newmodel = model.solver.add_output_units(len(train_dataset.classes))
            model.set_solver(newmodel, None)

        model.train_solver(None, all_data, None)
        result = model.test(None, test_datasets, verbose=verbose)
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

    jobs = []
    pool = mp.Pool()
    start = time.time()
    ntask = 10
    for task_order in range(ntask):
        
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
        save_order(result_folder, task_order, base_dataset.classes)

        
        traindata, testdata = base_dataset.train_test_split()

        dataset = traindata
        if args.oversampling:
            dataset = traindata.resampling()

        train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
        test_datasets, _, _ = testdata.split(tasks=2*args.tasks)
        
        print("CHECK")
        for t in test_datasets:
            print(t.pddata["ActivityName"].unique())

        print("******* Run ",task_order,"*******")
        print("\n")

        for method in methods:
            m, cmd = method
            identity["method"] = m
            args = copy.deepcopy(args)
            
            args.critic_fc_units = args.hidden_units
            args.generator_fc_units = args.hidden_units

            args.g_iters = get_g_iter(m, None)
            # run_model(identity, method, args, config, train_datasets, test_datasets, True)
            pool.apply_async(run_model, args=(identity, method, args, config, train_datasets, test_datasets, True))
            
    pool.close()
    pool.join()


    training_time = time.time() - start
    print(training_time)

    clearup_tmp_file(result_folder, ntask, methods)