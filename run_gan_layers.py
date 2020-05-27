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

def get_model_name(task_order, method, model_type, c):
    name = "t{task_order}-m{method}{c}-{type}.model".format(
            task_order=task_order,
            method=method,
            type=model_type,
            c=c)
    return name

def save_data_to_file(fpath, data, columns=None):
    data_loader = iter(utils.get_data_loader(data, 1))
    N = len(data)

    fo = open(fpath, "w")
    if columns is not None:
        fo.write(",".join(columns)+"\n")

    for x,y in data_loader:

        x = x.data.numpy()
        y = y
        
        s = [str(m) for m in x[0]]
        s.append(str(y[0]))
        
        fo.write(",".join(s)+"\n")
    
    fo.close()

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
        # ("sg-cgan", 0), 
        # ("sg-cgan", 1), 
        # ("sg-cgan", 2),
        ("sg-cgan", 3),
        # ("sg-cgan", 4)
    ]

    jobs = []
    # pool = mp.Pool()
    start = time.time()
    base_args = args

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
    
    traindata, testdata = base_dataset.train_test_split()

    dataset = traindata
    if args.oversampling:
        dataset = traindata.resampling()

    train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
    test_datasets, _, _ = testdata.split(tasks=args.tasks)

    # print("\n\nTraining Data")
    # for t in train_datasets:
    #     print(t, len(t), t.pddata["ActivityName"].unique())
    
    identity["task_order"] = 0
    for method in methods:
        m, cmd = method
        identity["method"] = m
        args = copy.deepcopy(base_args)
        
        args.critic_fc_units = select_hidden_unit(args, cmd)
        args.generator_fc_units = select_hidden_unit(args, cmd)
        args.solver_fc_units = select_hidden_unit(args, cmd)

        args.critic_fc_layers = cmd+3
        args.generator_fc_layers = cmd+3
        
        args.g_iters = get_g_iter(m, None)

        if args.output_model_path is None:
            model = run_model(identity, method, args, config, train_datasets, test_datasets, True)

            save_model(result_folder, identity, model.solver, "solver")
            if model.generator is not None:
                save_model(result_folder, identity, model.generator, "generator")
        else:
            output_model_path = args.output_model_path
            print("Load Models")
            cuda = torch.cuda.is_available()
            device = torch.device("cuda" if cuda else "cpu")

            solver = Classifier(
                input_feat=config['feature'],
                classes=config['classes'],
                fc_layers=args.solver_fc_layers, fc_units=args.solver_fc_units, 
                cuda=cuda,
                device=device,
            ).to(device)

            model_name = get_model_name(0, m, "solver", str(cmd))
            solver.load_model(output_model_path+model_name)

            generator = arg_params.get_generator(m, config, cuda, device, args, init_n_classes=config["classes"])
            model_name = get_model_name(0, m, "generator", str(cmd))
            generator.load_model(output_model_path+model_name, n_classes=config['classes'])
            model = GenerativeReplayLearner(args, classes_per_task, visdom=None)
            model.set_solver(solver)
            model.set_generator(generator)
            
            for c in dataset.classes:
                model.classmap.map(c)
        
        N = 1000
        
        def no_transform(y):
            return y
            
        generated_data = model.sample(model.classmap.classes, N, verbose=False)
        generated_data.set_target_tranform(no_transform)
        
        filename = get_model_name(0, m, "sample", str(cmd)).replace(".model", ".feat")

        save_data_to_file(result_folder+filename, generated_data, columns=dataset.pddata.columns)

    training_time = time.time() - start
    print(training_time)

    # clearup_tmp_file(result_folder, ntask, methods)