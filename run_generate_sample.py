#!/usr/bin/env python3
import torch
import numpy as np
from smart_home_dataset import SmartHomeDataset
from classifier import Classifier
from torch import optim
import utils
import callbacks as cb
import time
import math
from generative_replay_learner import GenerativeReplayLearner;
import arg_params
from torch.nn import functional as F
from scipy.stats import entropy
# import ot

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
        y = y.data.numpy()
        s = [str(m) for m in x[0]]
        s.append(str(y[0]))
        
        fo.write(",".join(s)+"\n")
    
    fo.close()
    
if __name__ == "__main__":

    parser = arg_params.get_parser()
    args = parser.parse_args()
    print("Arguments")
    print(args)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
        
    result_folder = args.results_dir

    print("\n")
    print("STEP1: load datasets")

    base_dataset = select_dataset(args)
    
    traindata, testdata = base_dataset.train_test_split()

    dataset = traindata

    if args.oversampling:
        dataset = traindata.resampling()
    
    
    train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
    
    
    base_args = args

    fout = open(result_folder+"gan_score.txt", "w")
    fout.write("task_order, method, n_real, n_fake, offline_acc_real, offline_acc_fake, is, is_err, mmd, knn_tp, knn_fp, knn_fn, knn_tn\n")

    fto = open(result_folder+"task_orders.txt")
    task_order = [line.strip().split(";") for line in fto]
    fto.close()
    
    N = int(len(base_dataset)/len(base_dataset.classes))
    # for t in range(len(task_order)):
    for t in [0]:
        
        for g in ["mp-gan", "mp-wgan", "sg-cgan", "sg-cwgan"]:
            solver = Classifier(
                input_feat=config['feature'],
                classes=config['classes'],
                fc_layers=args.solver_fc_layers, fc_units=args.solver_fc_units, 
                cuda=cuda,
                device=device,
            ).to(device)

            model_name = get_model_name(t, g, "solver", "0")
            solver.load_model(result_dir+model_name)

            generator = arg_params.get_generator(g, config, cuda, device, args, init_n_classes=config["classes"])
            model_name = get_model_name(t, g, "generator", "0")
            generator.load_model(result_dir+model_name, n_classes=config['classes'])



            model = GenerativeReplayLearner(args, classes_per_task, visdom=None)
            model.set_solver(solver)
            model.set_generator(generator)

            for c in dataset.classes:
                model.classmap.map(c)

            generated_data = model.sample(model.classmap.classes, N, verbose=False)
            generated_data.set_target_tranform(model.target_transform())
            

            filename = get_model_name(t, g, "sample", "0").replace(".model", ".feat")

            save_data_to_file(filename, generated_data, columns=dataset.pddata.columns)
            
            print("\t", g)

        print("Task Order", t, "DONE")

        fout.close()
        
    



    
    


