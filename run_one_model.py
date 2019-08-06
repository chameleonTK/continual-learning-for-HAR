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

if __name__ == '__main__':
    args = arg_params.get_args()

    args.iters = 1000
    args.g_iters = 1000
    args.g_log = int(args.g_iters*0.05)
    args.solver_fc_units = 500
    args.critic_fc_units = 100
    args.generator_fc_units = 100

    args.visdom = True
    args.self_verify = True
    args.oversampling = True
    args.solver_ewc = False
    args.solver_distill = True
    args.generator_noise = True

    # Use cuda?
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    print("\n")
    print("STEP1: load datasets")

    base_dataset = SmartHomeDataset(args.data_dir)

    traindata, testdata = base_dataset.train_test_split()
    dataset = traindata
    if args.oversampling:
        dataset = traindata.resampling()

    train_datasets, config, classes_per_task = traindata.split(tasks=args.tasks)
    test_datasets, _, _ = testdata.split(tasks=args.tasks)
    
    if args.visdom:
        env_name = "Continual learning with smart home data"
        visdom = {'env': env_name, 'graph': "models"}
    else:
        visdom = None

    print("\n")
    print("STEP2: initial a solver")

    solver = Classifier(
        input_feat=config['feature'],
        classes=len(train_datasets[0].classes),
        fc_layers=args.solver_fc_layers, fc_units=args.solver_fc_units, 
        cuda=cuda,
        device=device,
    ).to(device)
    # utils.print_model_info(solver, title="SOLVER")


    print("STEP3: initial a generator")
    generator = arg_params.get_generator("mp-gan", config, cuda, device, args, init_n_classes=classes_per_task)
    # utils.print_model_info(generator, title="GENERATOR")


    model = GenerativeReplayLearner(args, classes_per_task, visdom=visdom)
    model.set_solver(solver)
    model.set_generator(generator)
    
    print("\n")
    print("STEP4: train & test")
    
    start = time.time()
    prev_active_classes = []
    if args.generative_model is not None:
        model.generator.load_model(args.generative_model, n_classes=config['classes'])


    model.eval_cb = cb._task_loss_cb(model, test_datasets, log=args.log, visdom=visdom, iters_per_task=args.iters)

    all_data = None
    for task, train_dataset in enumerate(train_datasets, 1):
        if all_data is None:
            all_data = train_dataset
        else:
            all_data = all_data.merge(train_dataset)

    result = None
    if args.replay == "offline":
        for task, train_dataset in enumerate(train_datasets, 1):
            if task ==0:
                continue

            newmodel = model.solver.add_output_units(len(train_dataset.classes))
            model.set_solver(newmodel)
                
        model.train_solver(500, all_data, None)
        result = model.test(500, test_datasets)
    else: 

        prev_dataset = None
        for task, train_dataset in enumerate(train_datasets, 1):

            if task>1:

                model.generator.classes += len(train_dataset.classes)
                newmodel = model.solver.add_output_units(len(train_dataset.classes))
                model.set_solver(newmodel, model.solver)

            print("* Task", task)
            print("Data size:", len(train_dataset))

            active_classes_index = model.get_active_classes_index(task)
            print("Active classes:", active_classes_index)

            replayed_dataset = None
            if args.replay == "generative":
                replayed_dataset = model.sample(prev_active_classes, 500)
            elif args.replay == "exact":
                replayed_dataset = prev_dataset
                
            model.train_solver(task, train_dataset, replayed_dataset)
            if (args.generative_model is None) and (args.replay == "generative"):
                model.train_generator(task, train_dataset, replayed_dataset)

            if prev_dataset is None:
                prev_dataset = train_dataset
            else:
                prev_dataset = prev_dataset.merge(train_dataset)

            prev_active_classes = model.classmap.classes
            
            print("\n")
            result = model.test(task, test_datasets)
            print("\n\n")

    training_time = time.time() - start
    print("Training Time: ", training_time)


    print("STEP5: load and test")
    if (args.generative_model is None) and args.output_model_path is not None:
        model.generator.save_model(args.output_model_path)
    
    if args.evaluate:
        print("STEP5: evaluating")
        print("\n\n")
        print("Offline model")
        model.train_solver(500, all_data, None)
        offline = model.test(500, test_datasets)


        acc = 0
        for i, v in enumerate(offline["Precision"]):
            acc += abs(offline["Precision"][i]-result["Precision"][i])

        acc /= len(offline["Precision"])
        print("Average Differences:", acc*100, "%")