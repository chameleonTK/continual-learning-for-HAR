from torch.utils.data import Dataset
import numpy as np
# import main as env
import math
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

class SmartHomeDataset(Dataset):
    def __init__(self, filename, random_state = 10, rawdata = None, classes = None):
        if rawdata is None:
            self.pddata = pd.read_csv(filename).dropna()
        else:
            self.pddata = rawdata

        # self.classes = self.pddata["ActivityName"].unique()
        if classes is None:    
            self.classes = [
                "R1_work_at_computer",
                "R2_work_at_computer",
                "R1_sleep",
                "R2_sleep",
                "R1_bed_to_toilet",
                "R2_bed_to_toilet",
                # "R1_groom",
                # "R2_groom",
                # "R1_breakfast",
                # "R2_breakfast",

                "R2_prepare_dinner",
                "R2_watch_TV",
                
                "R2_prepare_lunch",
                "R1_work_at_dining_room_table",
                # "Cleaning",
                # "Wash_bathtub",
            ]
        else:
            self.classes = classes

        self.pddata = self.pddata[self.pddata['ActivityName'].isin(self.classes)]
        self.labels = self.pddata["ActivityName"].values
        self.arr = self.pddata.drop('ActivityName', 1).values
        
        self.random_state = random_state
        

        self.config = {
            'feature': len(self.pddata.columns) - 1, 
            # 'size': int(math.sqrt(len(self.pddata.columns) - 1)), 
            # 'channels': 1, 
            'classes': len(self.classes)
        }
        

        self.target_tranform = None

        
    def permu_task_order(self, classes = None):
        if classes is not None:
            self.classes = classes
        else:
            lst = np.random.permutation(self.classes)
            self.classes = lst[0:10]

        print("Task order: ", self.classes)
        return self.classes
        

    def train_test_split(self):        
        traindata, testdata = train_test_split(self.pddata, test_size=0.2, random_state=self.random_state)

        traindata = SmartHomeDataset("", rawdata=traindata, classes=self.classes)
        testdata = SmartHomeDataset("", rawdata=testdata, classes=self.classes)

        return (traindata, testdata)

    def set_target_tranform(self, transform):
        self.target_tranform = transform

    def __len__(self):
        return len(self.pddata)

    def __getitem__(self, idx):
        if self.target_tranform is not None:
            return torch.FloatTensor(list(self.arr[idx])), self.target_tranform(self.labels[idx])
            
        
        raise Exception("No key `"+self.labels[idx]+"` in classmap")
        # return torch.FloatTensor(list(self.arr[idx])), self.labels[idx]
    
    def __str__(self):
        return "Smart Home dataset"

    def resampling(self):

        # SMOTE
        # It will synthesize a new data for the minority class, based on those that already exist.

        # Tomek links
        # It pairs an instance with another instance but of opposite classes 
        # and then remove the instances of the bigger class of each pair.

        smote = SMOTE(random_state=self.random_state)


        classes = self.classes
        mapping = dict(zip(classes,range(len(classes))))

        pddata = self.pddata.replace({'ActivityName': mapping})
        labels = pddata["ActivityName"]
        
        
        pddata = pddata.drop('ActivityName', 1)
        columns = pddata.columns

        X_sm, y_sm = smote.fit_sample(pddata, labels)
        

        # convert the output back to pandas
        X_sm = pd.DataFrame(data=X_sm, columns=columns)
        rmapping = dict(zip(range(len(classes)), classes))
        X_sm["ActivityName"] = y_sm
        X_sm = X_sm.replace({'ActivityName': rmapping})

       
        return SmartHomeDataset("", rawdata=X_sm, classes=self.classes)

    def split(self, tasks=5):

        classes_per_task = int(np.floor(len(self.classes) / tasks))
        datasets = []
    
        labels_per_task = [
            list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
        ]

        # split them up into sub-tasks
        datasets = []
        for labels in labels_per_task:
            datasets.append(self.filter(labels))

        return datasets, self.config, classes_per_task

    def filter(self, labels):
        classes = [self.classes[l] for l in labels]
        filtered = self.pddata[self.pddata["ActivityName"].isin(classes)]

        return SmartHomeDataset("", rawdata=filtered, classes=classes)

    def merge(self, dataset):
        mpddata = pd.concat([self.pddata, dataset.pddata], ignore_index=True)
        mpddata = mpddata.sample(frac=1).reset_index(drop=True)
        classes = self.classes + dataset.classes
        return SmartHomeDataset("", rawdata=mpddata, classes=classes)

    def detail(classes, train_datasets, test_datasets):
        print ("\n\n===== Smart Home dataset ======")
        d = {'Activity Name': [], '#Train': [], "#Test": [], "Total":[]}
        
        cid = {}
        i = 0
        for c in classes:
            d["Activity Name"].append(c)
            d["#Train"].append(0)
            d["#Test"].append(0)
            d["Total"].append(0)

            cid[c] = i
            i += 1

        for dataset in train_datasets:
            data = dataset.pddata
            for c in classes:
                tr = data[data["ActivityName"] == c]
                d["#Train"][cid[c]] += len(tr)

                d["Total"][cid[c]] += len(tr)
        
        for dataset in test_datasets:
            data = dataset.pddata
            for c in classes:
                tr = data[data["ActivityName"] == c]
                d["#Test"][cid[c]] += len(tr)
                d["Total"][cid[c]] += len(tr)

        df = pd.DataFrame(data=d)
        print (df)
