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

def clearup_tmp_file(result_folder, ntask, methods, delete=True):

    fresult = open(result_folder+"results.txt", "w")
    fresult.write("task_order, method, cmd, train_session, task_index, no_of_test, no_of_correct_prediction, accuracy, solver_training_time, generator_training_time\n")
 
    for task_order in range(ntask):
        for method in methods:
            m, cmd = method

            fname = "_t{task_order}-m{method}{c}_results.tmp".format(
                task_order=task_order,
                method=m,
                c=str(cmd))

            try:
                fo = open(result_folder+fname)
                for line in fo:
                    fresult.write(line)
                fo.close()

                if delete:
                    os.remove(result_folder+fname)
            except Exception as e:
                print(e)

    fresult.close()


if __name__ == "__main__":

    parser = arg_params.get_parser()
    args = parser.parse_args()
    print("Arguments")
    print(args)

    result_folder = args.results_dir

    methods = [ 
        ("offline", 0), ("none", 0), ("exact", 0), ("mp-gan", 0), ("mp-wgan", 0), ("sg-cgan", 0), ("sg-cwgan", 0), ("lwf", 0), ("ewc", 0),
        ("offline", 1), ("none", 1), ("exact", 1), ("mp-gan", 1), ("mp-wgan", 1), ("sg-cgan", 1), ("sg-cwgan", 1), ("lwf", 1), ("ewc", 1),
        ("offline", 2), ("none", 2), ("exact", 2), ("mp-gan", 2), ("mp-wgan", 2), ("sg-cgan", 2), ("sg-cwgan", 2), ("lwf", 2), ("ewc", 2),
        ("offline", 3), ("none", 3), ("exact", 3), ("mp-gan", 3), ("mp-wgan", 3), ("sg-cgan", 3), ("sg-cwgan", 3), ("lwf", 3), ("ewc", 3),
    ]

    start = time.time()
    ntask = 10


    training_time = time.time() - start
    print(training_time)

    clearup_tmp_file(result_folder, ntask, methods, delete=False)
