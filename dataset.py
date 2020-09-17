from torch.utils.data import Dataset
import pandas as pd 
import numpy as np 
import torch

class Mydataset(Dataset):


    def __init__(self):
        self.path="~/code/math/excel_to_python.csv"
        data = np.array(pd.read_csv(self.path,sep='\t'));
        self.char= data[:,1:2].reshape(-1,50,1)
        self.index = data[:,0:1].reshape(-1,50,1)
        train_data=  data[:,2:22]
        test_data = data[:,22:23]
        train_data = train_data.reshape(-1,50,22)
        test_data = test_data.reshape(-1,50,1)
        self.train_data = torch.from_numpy(train_data);
        self.test_data =  torch.from_numpy(test_data)
        self.len = data.shape[0]


        # super().__init__()

    def __getitem__(self, index):
        return self.train_data[index],self.test_data[index],self.char[index],self.index[index]

        # return super().__getitem__(index)


    def __len__(self):
        return self.len