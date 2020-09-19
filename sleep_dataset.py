from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np 
import torch

class sleepDataset(Dataset):

    def __init__(self):
        self.path="data\sleep_data1.txt"
        data = np.array(pd.read_csv(self.path,sep='\t'))

        train_data = data[:,1:5] / 100.0
        label_data = data[:, 0] - 2
        label_data.reshape(-1, 1)

        self.train_data = torch.from_numpy(train_data)
        self.label_data = torch.from_numpy(label_data)
        self.len=data.shape[0]


    def __getitem__(self,index):
        return self.train_data[index],self.label_data[index]


    def __len__(self):
        return self.len

if __name__ == "__main__":
    dataset = sleepDataset() 
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=10)

    for batch, (x, y) in enumerate(loader):
        print(x)
        break