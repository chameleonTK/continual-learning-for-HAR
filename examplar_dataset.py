import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch

import pandas as pd 

class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, prev_active_classes):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.classes = []
        # testdata = SmartHomeDataset("", rawdata=testdata, classes=self.classes)
        if len(self.exemplar_sets) == 0:
            self.pddata = pd.DataFrame([])
        else:
            columns = list(range(self.exemplar_sets[0].shape[1]))

            self.pddata = pd.DataFrame([], columns=columns)
            for class_id in range(len(self.exemplar_sets)):
                dataset = self.exemplar_sets[class_id]
                df = pd.DataFrame(dataset)
                df["ActivityName"] = prev_active_classes[class_id]

                self.pddata = pd.concat([self.pddata, df], axis=0, ignore_index=True)

            

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)