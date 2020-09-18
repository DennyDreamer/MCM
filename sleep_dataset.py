from torch.utils.data import Dataset
import pandas as pd 
import numpy as np 
import torch

class sleepDataset(Dataset):

    def __init__(self):
        self.path="sleep_data.txt"
        data = np.array(pd.read_csv(self.path,sep='\t'))

        train_data = data[:,1:5]
        label_data = data[:,0]

        self.train_data = torch.from_numpy(train_data)
        self.label_data = torch.from_numpy(label_data)
        self.len=data.shape[0]


    def __getitem__(self,index):
        return self.train_data[index],self.label_data[index]


    def __len__(self):
        return self.len