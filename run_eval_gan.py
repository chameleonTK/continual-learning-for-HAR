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

# got this function from https://github.com/sbarratt/inception-score-pytorch
def inception_score(generated_data, model, batch_size=5):
    data_loader = iter(utils.get_data_loader(generated_data, batch_size))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    N = len(generated_data)

    model.eval()
    preds = np.zeros((len(generated_data), len(generated_data.classes)))

    i=0
    for x,y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            scores = model(x)
            scores = F.softmax(scores, dim=1)
         
            scores = scores.data.cpu().numpy()
            batch_size_i = x.size()[0]
            preds[i*batch_size:i*batch_size + batch_size_i] = scores
        
        i+=1

    splits = 10
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))

        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# adopted code from https://github.com/xuqiantong/GAN-Metrics
def distance(X, Y, sqrt):

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    data_loaderX = iter(utils.get_data_loader(X, len(X)))
    data_loaderY = iter(utils.get_data_loader(Y, len(Y)))
    X, _ = next(data_loaderX)
    Y, _ = next(data_loaderY)
    
    X, Y = X.to(device), Y.to(device)
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M

# def wasserstein(M, sqrt):
#     if sqrt:
#         M = M.abs().sqrt()
#     emd = ot.emd2([], [], M.numpy())

#     return emd


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)

    if sqrt:
        M = M.abs().sqrt()

    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    tp = (pred * label).sum()
    fp = (pred * (1 - label)).sum()
    fn = ((1 - pred) * label).sum()
    tn = ((1 - pred) * (1 - label)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    acc_real = tp / (tp + fn)
    acc_fake = tn / (tn + fp)
    acc = torch.eq(label, pred).float().mean()
    k = k

    return tp, fp, fn, tn, precision, recall, acc_real, acc_fake, acc, k


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd

def accuracy(data, model):
    data_loader = iter(utils.get_data_loader(data, 5))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    N = len(data)

    model.eval()
    preds = np.zeros((len(generated_data), len(generated_data.classes)))

    total_correct = 0
    total_tested = 0
    for x,y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            scores = model(x)
            scores = F.softmax(scores, dim=1)
         
            _, predicted = torch.max(scores, 1)
                
                
        total_correct += (predicted == y).sum().item()
        total_tested += len(x)
    
    return float(total_correct) / float(total_tested)

def get_model_name(task_order, method, model_type, c):
    name = "t{task_order}-m{method}{c}-{type}.model".format(
            task_order=task_order,
            method=method,
            type=model_type,
            c=c)
    return name
            
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
    
    for t in range(len(task_order)):
        testdata.classes = task_order[t]
        print("Task Order", t)
        test_datasets, _, _ = testdata.split(tasks=args.tasks)

        assert (train_datasets[0].classes==task_order[t][0:2])

        offline_solver = Classifier(
            input_feat=config['feature'],
            classes=config['classes']+2,
            fc_layers=args.solver_fc_layers, fc_units=args.solver_fc_units, 
            cuda=cuda,
            device=device,
        ).to(device)


        
        model_name = get_model_name(t, "offline", "solver", "0")
        offline_solver.load_model(result_dir+model_name)

        offline_model = GenerativeReplayLearner(args, classes_per_task, visdom=None)
        offline_model.set_solver(offline_solver)
        for c in dataset.classes:
            offline_model.classmap.map(c)

        # result = offline_model.test(None, test_datasets, verbose=False)


        all_data = None
        for task, train_dataset in enumerate(train_datasets, 1):
            if all_data is None:
                all_data = train_dataset
            else:
                all_data = all_data.merge(train_dataset)

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
            all_data.set_target_tranform(model.target_transform())

            fake_acc = accuracy(generated_data, offline_solver)
            real_acc = accuracy(all_data, offline_solver)

            is_score, is_err = inception_score(generated_data, model.solver)

            Mxx = distance(all_data, all_data, False)
            Mxy = distance(all_data, generated_data, False)
            Myy = distance(generated_data, generated_data, False)
            # wasserstein(Mxy, True)
            mmd_score = mmd(Mxx, Mxy, Myy, 1)
            knn_score = knn(Mxx, Mxy, Myy, 1, False)
            knn_tp, knn_fp, knn_fn, knn_tn = knn_score[0:4]

            resp = [
                str(t),
                g,
                str(len(all_data)),
                str(len(generated_data)),
                str(real_acc),
                str(fake_acc),
                str(is_score),
                str(is_err),
                str(mmd_score),str(float(knn_tp)),
                str(float(knn_fp)),
                str(float(knn_fn)),
                str(float(knn_tn))
            ]
            
            fout.write(",".join(resp)+"\n")
            
            print("\t", g)

        print("Task Order", t, "DONE")

        fout.close()
        
    



    
    


