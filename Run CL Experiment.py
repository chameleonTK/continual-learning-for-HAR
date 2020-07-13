#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import gc

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# # Load Data and code

# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[3]:


# cd "/content/drive/My Drive/continual-learning"


# In[4]:


# !git clone https://github.com/chameleonTK/continual-learning-for-HAR.git src


# In[5]:


# cd "src"


# In[6]:


# !pip install visdom


# In[7]:


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


# # Get parameters

# In[17]:


params = {
    "--results-dir": "./Results/CASAS/",
    "--data-dir": "casas",
    "--task-order": "./Results/CASAS/task_orders.txt",
    "--batch": 1000
}

p = [f"{k} {params[k]}" for k in params]
p = " ".join(p)


# In[18]:


parser = arg_params.get_parser()
args = parser.parse_args(p.split())
print("Arguments")
for attr, value in args.__dict__.items():
    print("  * ", attr, value)

args.visdom = True
result_folder = args.results_dir


# In[ ]:





# # Get Dataset

# In[19]:


def select_dataset(args, classes=None):
    if args.data_dir == "pamap":
        default_classes = [
            'lying', 
            'sitting', 
            'standing', 
            'ironing', 
            'vacuum cleaning', 
            'ascending stairs', 
            'walking', 
            'descending stairs', 
            'cycling', 
            'running'
        ]

        data_dir = "./Dataset/PAMAP2/pamap.feat"

    elif args.data_dir == "dsads":
        default_classes = [
            "sitting",
            "standing",
            "lying on back side",
            "lying on right side",
            "ascending stairs",
            "descending stairs",
            "exercising on a stepper",
            "rowing",
            "jumping",
            "playing basketball"
        ]

        data_dir = "./Dataset/DSADS/dsads.feat"


    elif args.data_dir == "housea":
        default_classes = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'] #skip A6
        args.tasks = 3
        data_dir = "./Dataset/House/HouseA.feat"

    elif args.data_dir == "casas":
        default_classes = [
            "R1_work_at_computer",
            "R2_work_at_computer",
            "R1_sleep",
            "R2_sleep",
            "R1_bed_to_toilet",
            "R2_bed_to_toilet",

            "R2_prepare_dinner",
            "R2_watch_TV",
            
            "R2_prepare_lunch",
            "R1_work_at_dining_room_table",
        ]
        data_dir = "./Dataset/twor.2009/annotated.feat.ch1"
    else:
        raise Exception("Unknow dataset")
    
    if classes is None:
        classes = default_classes
        
    return SmartHomeDataset(data_dir, classes=classes)


# In[20]:


tasks = []
if args.task_order is not None:
    ft = open(args.task_order)
    tasks = [line.strip().split(";") for line in ft]


# In[21]:


# tasks


# # Train a model

# In[22]:



def result_to_list(identity, results):
    lst = []
    for idx, session in enumerate(results["Task"]):
        lst.append([
            identity["task_order"],
            identity["method"],
            identity["cmd"],
            identity["train_session"],
            results["Task"][idx],
            results["#Test"][idx],
            results["#Correct"][idx],
            results["Accuracy"][idx],
            identity["solver_training_time"],
            identity["generator_training_time"],
        ])

    return lst

def save_results(result_folder, identity, results):

    fname = "_t{task_order}-m{method}{c}_results.csv".format(
        task_order=identity["task_order"],
        method=identity["method"],
        c=identity["cmd"])
    
    o = None
    for r in results:
        if o is None:
            o = r
        else:
            o = o + r

    df = pd.DataFrame(o, columns=[
        "Task Sequence Idx", 
        "Method",
        "Method Options",
        "Training Session",
        "Task",
        "#Test",
        "#Correct",
        "Accuracy",
        "Solver Training Time",
        "Generator Training Time",
    ])
    
    df.to_csv(result_folder+fname, index=False)
    


# In[23]:



def run_model(identity, method, args, config, train_datasets, test_datasets, verbose=False, visdom=None):
    #try:   
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
            args.solver_ewc = False
            
        elif m== "ewc":
            args.solver_distill = False
            args.solver_ewc = True
            
        elif m in ["none", "exact", "offline"]:
            args.solver_distill = False
            args.solver_ewc = False

        identity["cmd"] = str(cmd)
        
        for attr, value in args.__dict__.items():
            if attr in []:
                print("  * ", attr, value)
        

        model = GenerativeReplayLearner(args, 2, verbose=verbose, visdom=visdom)
        
        if visdom is not None:
            model.eval_cb = cb._task_loss_cb(model, test_datasets, log=args.log, visdom=visdom, iters_per_task=args.iters)
            
        solver = Classifier(
            input_feat=config['feature'],
            classes=len(train_datasets[0].classes),
            fc_layers=args.solver_fc_layers, fc_units=args.solver_fc_units, 
            cuda=cuda,
            device=device,
        ).to(device)
        model.set_solver(solver)

    
        if m in ["mp-gan", "mp-wgan", "sg-cgan", "sg-cwgan"]:
            args.replay = "generative"

            generator = arg_params.get_generator(m, config, cuda, device, args, init_n_classes=2)
            model.set_generator(generator)

            args.g_log = int(args.g_iters*0.05)

        else:
            generator = None
        
        if args.replay == "offline":
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
        else: 

            prev_active_classes = []
            prev_dataset = None
            for task, train_dataset in enumerate(train_datasets, 1):
                identity["train_session"] = task

                if task>1:
                    if model.generator is not None:
                        model.generator.classes += len(train_dataset.classes)
                        
                    newmodel = model.solver.add_output_units(len(train_dataset.classes))
                    model.set_solver(newmodel, model.solver)

                active_classes_index = model.get_active_classes_index(task)

                replayed_dataset = None
                if args.replay == "generative":

                    if args.replay_size <= 1:
                        # when replay_size in [0, 1]; # samples == replay_size * len(train_dataset)
                        replayed_dataset = model.sample(prev_active_classes, 2*len(train_dataset), n=args.replay_size*len(train_dataset))
                    else:
                        # otherwise; #samples == replay_size * len(active_classes_index)
                        replayed_dataset = model.sample(prev_active_classes, args.replay_size)


                elif args.replay == "exact":
                    replayed_dataset = prev_dataset
                
                start = time.time()
                model.train_solver(task, train_dataset, replayed_dataset, rnt=args.rnt)
                training_time = time.time() - start

                identity["solver_training_time"] = training_time

                start = time.time()
                if (args.generative_model is None) and (args.replay == "generative"):
                    model.train_generator(task, train_dataset, replayed_dataset)
                    
                training_time = time.time() - start

                identity["generator_training_time"] = training_time

                if prev_dataset is None:
                    prev_dataset = train_dataset
                else:
                    prev_dataset = prev_dataset.merge(train_dataset)

                prev_active_classes = model.classmap.classes
                
                result = model.test(task, test_datasets, verbose=verbose)
                results.append(result_to_list(identity, result))


            
        # save_model(result_folder, identity, model.solver, "solver")
        # if model.generator is not None:
        #     save_model(result_folder, identity, model.generator, "generator")
        

        save_results(result_folder, identity, results)
        
        return model, results
    
    #except Exception as e:
    #    print("ERROR:", e)

    #print("DONE task order", identity["task_order"])
    


# In[24]:


methods = [ 
    # ("offline", 0), 
    # ("none", 0), 
    # ("exact", 0), 
    ("mp-gan", 0), ("mp-wgan", 0), ("sg-cgan", 0), ("sg-cwgan", 0), 
#     ("lwf", 0), ("ewc", 0)
]


# In[25]:




start = time.time()
for task_order, classes in enumerate(tasks):
    
    print(f"=== IDX {task_order} ===")
    base_dataset = select_dataset(args, classes)
    
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
    # if args.oversampling:
    #     dataset = traindata.resampling()

    train_datasets, config, classes_per_task = dataset.split(tasks=args.tasks)
    test_datasets, _, _ = testdata.split(tasks=args.tasks)

    # Check distribution of label
    # for d in dataset.pddata["ActivityName"].unique():
    #     x = dataset.pddata
    #     print(d, len(x[x["ActivityName"]==d]))

    print("******* Run ", task_order, "*******")
    print("\n")

    base_args = args
    for method in methods:
        m, cmd = method
        identity["method"] = m
        args = copy.deepcopy(base_args)

        visdom = {'env': f"Method: {m}, options: {cmd}", 'graph': "models", "values":[], "gan_loss": {}}
        model, results = run_model(identity, method, args, config, train_datasets, test_datasets, True, visdom=visdom)
        
        training_time = time.time() - start
        print("")
        print(f"Training Time[{m}]:", training_time)
    break

training_time = time.time() - start
print("Overall Training Time:", training_time)
print()



