import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import utils
import callbacks as cb
import tqdm
import copy
import pandas as pd 
from smart_home_dataset import SmartHomeDataset


class GenerativeReplayLearner():

    def __init__(self, args, classes_per_task, visdom=None, verbose=True):
        self.solver = None
        self.generator = None
        self.visdom = visdom
        self.previous_solver = None
        self.previous_generator = None
        self.args = args
        self.classes_per_task = classes_per_task
        self.eval_cb = None
        self.classmap = ClassMap()
        self.verbose = verbose


        
        self.solver_ewc = args.solver_ewc
        self.solver_distill = args.solver_distill

        self.generator_noise = args.generator_noise

    
    def set_solver(self, solver, previous_solver=None):
        self.previous_solver = previous_solver
        self.solver = solver
        self.solver.ewc = self.solver_ewc
        self.solver.distill = self.solver_distill
        
        self.solver_loss_cbs = [
            cb._solver_loss_cb(
                log=self.args.log,
                model=solver,
                visdom=self.visdom,
                tasks=self.args.tasks,
                iters_per_task=self.args.iters,
                progress_bar=self.verbose)
        ]

    def set_generator(self, generator):
        self.generator = generator
        self.generator.noisy = self.generator_noise

        self.generator_loss_cbs = [
            cb._generator_training_callback(
                log=self.args.g_log,
                model=generator,
                visdom=self.visdom,
                tasks=self.args.tasks,
                iters_per_task=self.args.g_iters,
                progress_bar=self.verbose)
        ]

    def get_active_classes_index(self, task):
        return self.classmap.class_index()
        # return list(range(self.classes_per_task * task))

    def target_transform(self):
        vm = self
        def cb(y):
            return vm.classmap.map(y)

        return cb

    def train_solver(self, task, train_dataset, replayed_dataset = None, rnt=0.5):
        print("=> Train Solver")
        model = self.solver
        iters = self.args.iters
        batch_size = self.args.batch
        loss_cbs = self.solver_loss_cbs
        classes_per_task = self.classes_per_task
        

        prev_active_classes = self.get_active_classes_index(task)
        # regis new class to classmap
        for c in train_dataset.classes:
            self.classmap.map(c)

        train_dataset.set_target_tranform(self.target_transform())

        if replayed_dataset is not None:
            replayed_dataset.set_target_tranform(self.target_transform())

        

        # Set model in training-mode
        model.train()
        previous_model = None

        # Use cuda?
        cuda = model._is_on_cuda()
        device = model._device()
        active_classes = self.get_active_classes_index(task)
        print(active_classes)
        
        
        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        prev_total_loss = float('inf')

        for epoch in range(1, iters+1):

            data_loader = iter(utils.get_data_loader(train_dataset, min(len(train_dataset), batch_size), cuda=cuda, drop_last=True))
            if (replayed_dataset is not None):
                replayed_data_loader = iter(utils.get_data_loader(replayed_dataset, min(len(replayed_dataset), batch_size), cuda=cuda, drop_last=True))

            assert(len(data_loader) >= 1)

            total_loss = 0
            for batch_index in range(1, len(data_loader)+1):
                x, y = next(data_loader)
                x, y = x.to(device), y.to(device)
                
                scores = None
                if self.previous_solver is not None:
                    
                    with torch.no_grad():
                        scores = self.previous_solver(x)
                        scores = scores.cpu()
                        scores = scores[:, prev_active_classes]


                x_ = None
                y_ = None
                scores_ = None

                if replayed_dataset is not None:
                    try:
                        x_, y_ = next(replayed_data_loader)                               
                        x_, y_ = x_.to(device), y_.to(device)
                    except StopIteration:
                        continue

                    if self.previous_solver is not None:
                        with torch.no_grad():
                            scores_ = self.previous_solver(x_)
                            scores_ = scores_.cpu()
                            scores_ = scores_[:, prev_active_classes]
                            
                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt=rnt)

                
                if self.eval_cb is not None:
                    self.eval_cb(batch_index, task=task)
                
                total_loss += loss_dict["loss_total"]

            
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, epoch, {
                        "loss_total": total_loss, 
                        "accuracy": loss_dict["accuracy"]
                    }, task=task)

            if epoch % 50 == 0:
                if prev_total_loss < total_loss or total_loss < 0.001:
                    print("Early stopping")
                    break

                prev_total_loss = total_loss


        # Close progres-bar(s)
        progress.close()
        previous_model = copy.deepcopy(model).eval()
        self.previous_solver = previous_model
        
        if self.solver_ewc:
            model.estimate_fisher(train_dataset, allowed_classes=active_classes)

    def train_generator(self, task, train_dataset, replayed_dataset = None):
        print("=> Train Generator")
        model = self.generator
        iters = self.args.g_iters
        batch_size = self.args.batch
        loss_cbs = self.generator_loss_cbs
        target_transform = self.target_transform()

        # Set model in training-mode
        model.train()

        # Use cuda?
        cuda = model._is_on_cuda()
        device = model._device()
        
        model._run_train(train_dataset, iters, batch_size, loss_cbs, target_transform, replayed_dataset=replayed_dataset)

    def test(self, task, test_datasets, verbose=True):
        print("=> Test")
        solver = self.solver

        cuda = solver._is_on_cuda()
        device = solver._device()
        active_classes = self.get_active_classes_index(task)
        solver.eval()
        
        d = {'Task': [], "#Test": [], "#Correct":[], 'Accuracy': []}
        for t, dataset in enumerate(test_datasets, 1):
            if (task is not None) and (t > task):
                break
            
            dataset.set_target_tranform(self.target_transform())
            data_loader = iter(utils.get_data_loader(dataset, min(50, len(dataset)), cuda=cuda, drop_last=True))
            total_tested = total_correct = 0
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)  
                with torch.no_grad():
                    scores = solver(data)
                    scores = scores[:, active_classes]
                    _, predicted = torch.max(scores, 1)
                
                
                total_correct += (predicted == labels).sum().item()
                total_tested += len(data)

            accuracy = total_correct / total_tested

            d["Task"].append(t)
            d["#Test"].append(total_tested)
            d["#Correct"].append(total_correct)
            d["Accuracy"].append(accuracy)
            

        if verbose:
            df = pd.DataFrame(data=d)
            print (df)
        
        return d

    def _verify(self, x_, y_, active_classes_index):
        device = self.generator._device()

        mode = self.solver.training
        self.solver.eval()

        x_ = x_.to(device)
        y_ = y_.to(device)
        y_hat = self.solver(x_)
        y_hat = y_hat[:, active_classes_index]
        y_hat = y_hat.to(device)

        mask = (y_ == y_hat.max(1)[1])
        x_ = x_[mask]

        self.solver.train(mode=mode)
        return x_

    def _sample(self, class_index, sample_size):
        x_ = self.generator.sample(class_index, sample_size)
        y_ = torch.LongTensor(list(np.full((sample_size, ), int(class_index))))

        device = self.generator._device()
        x_ = x_.to(device)
        y_ = y_.to(device)
        return (x_, y_)

    def sample(self, active_classes, sample_size, verbose=True, n=None):

        device = self.generator._device()
        all_samples = None
        all_labels = None
        
        active_classes_index = range(len(active_classes))
        for class_index, class_label in enumerate(active_classes, 0):
            
            # Iterate until it gets a proper number of samples
            x_ = None
            for i in range(10):
                tmpx_, tmpy_ = self._sample(class_index, sample_size)
                tmpx_= tmpx_.narrow(0, 0, int(sample_size*0.8))
                tmpy_ = tmpy_.narrow(0, 0, int(sample_size*0.8))
                # Self-verify
                if self.args.self_verify:
                    tmpx_ = self._verify(tmpx_, tmpy_, active_classes_index)
                    if len(tmpx_) ==0 and verbose:
                        print("WARNING: your generator cannot generate class"+str(class_index)+" properly")
                        # raise Exception("WARNING: your generator cannot generate class"+str(class_index)+" properly")

                if x_ is None:
                    x_ = tmpx_
                else:
                    x_ = torch.cat([x_, tmpx_], dim=0)
                    x_ = x_.to(device)
                    
                if len(x_) > sample_size:
                    x_ = x_.narrow(0, 0, sample_size)
                    break
                elif len(x_) == sample_size:
                    break
                   
            # If not, then uses just what it can do
            if len(x_) < sample_size:
                tmpx_, tmpy_ = self._sample(class_index, sample_size - len(x_))
                x_ = torch.cat([x_, tmpx_], dim=0)
                x_ = x_.to(device)
                
                if verbose:
                    print("WARNING: low quality sample on ["+str(class_index)+"]", len(tmpx_))

            print("=> class", active_classes[class_index], len(x_))
            y_ = np.full((len(x_), ), class_label)
            x_ = x_.cpu().data.numpy()

            
                
            if all_samples is None or len(all_samples)==0:
                all_samples = x_
                all_labels = y_
            else:
                all_samples = np.concatenate((all_samples, x_), axis=0)
                all_labels = np.concatenate((all_labels, y_), axis=0)

        
        if all_samples is None:
            return None
        
        df = pd.DataFrame(all_samples)
        df["ActivityName"] = all_labels
        df = df.sample(frac=1).reset_index(drop=True)

        if n is not None and n < len(df):
            df = df.sample(n=n).reset_index(drop=True)
        
        return SmartHomeDataset("", rawdata=df, classes=active_classes)


class ClassMap():
    def __init__(self):
        self.dict = {}
        self.classes = []

    def class_index(self):
        return [c for c in range(len(self.classes))]

    def map(self, p):
        if p in self.dict:
            return self.dict[p]
        else:
            self.dict[p] = len(self.classes)
            self.classes.append(p)
            return self.dict[p]
