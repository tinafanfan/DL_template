import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn import datasets


class IrisDataset(Dataset):

    # data loading
    def __init__(self):
        iris = datasets.load_iris()
        feature = pd.DataFrame(iris.data, columns=iris.feature_names)
        target = pd.DataFrame(iris.target, columns=['target'])
        iris_data = pd.concat([target, feature], axis=1)
        
        # keep only Iris-Setosa and Iris-Versicolour classes
        iris_data = iris_data[iris_data.target <= 1]
        self.x = torch.from_numpy(np.array(iris_data)[:, 1:])
        self.y = torch.from_numpy(np.array(iris_data)[:, [0]])
        self.y = torch.squeeze(self.y) # reduce dimension

        self.x = self.x.to(torch.float32)
        self.y = self.y.to(torch.float32)

        self.n_samples = self.x.shape[0]

    # working for indexing
    def __getitem__(self, index):
        
        return self.x[index], self.y[index]

    # return the length of our dataset
    def __len__(self):

        return self.n_samples

