from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import pywt
import torch

class LSTMdataset(Dataset):
    

    def __init__(self):
        self.path="data\data1_32_norm.txt"
        data = np.array(pd.read_csv(self.path,sep='\t'));
        self.char= data[:,1:2].reshape(-1,32,1)
        self.index = data[:,0:1].reshape(-1,32,1)
        train_data=  data[:,2:22]
        test_data = data[:,22:23]
        train_data = train_data.reshape(-1,32,20)
        test_data = test_data.reshape(-1,32,1)
        self.train_data = torch.from_numpy(train_data);
        self.test_data =  torch.from_numpy(test_data)
        self.len = data.shape[0]


        # super().__init__()

    def __getitem__(self, index):
        return self.train_data[index],self.test_data[index][0],self.char[index],self.index[index]

        # return super().__getitem__(index)


    def __len__(self):
        return len(self.train_data)


class Mydataset(Dataset):


    def __init__(self, path):
        self.path= path
        data = np.array(pd.read_csv(self.path,sep='\t'));
        self.char= data[:,1:2].reshape(-1,32,1)
        self.index = data[:,0:1].reshape(-1,32,1)
        train_data=  data[:,2:22]
        test_data = data[:,22:23]
        train_data = train_data.reshape(-1,1,32,20)
        test_data = test_data.reshape(-1,32,1)
        self.train_data = torch.from_numpy(train_data);
        self.test_data =  torch.from_numpy(test_data)
        self.len = data.shape[0]


        # super().__init__()

    def __getitem__(self, index):
        return self.train_data[index],self.test_data[index][0],self.char[index],self.index[index]

        # return super().__getitem__(index)


    def __len__(self):
        return len(self.train_data)

if __name__ == "__main__":
    path="data1.txt"
    data = np.array(pd.read_csv(path,sep='\t'));
    char= data[:,1:2].reshape(-1,128,1)
    index = data[:,0:1].reshape(-1,128,1)
    train_data =  data[:,2:22]
    test_data = data[:,22:23]
    train_data = train_data.reshape(-1,1,128,20)
    test_data = test_data.reshape(-1,128,1)

    x = torch.from_numpy(train_data)
    y =  torch.from_numpy(test_data)
    x = F.batch_norm(x, )

    w = pywt.Wavelet('db18')
    maxlev = pywt.dwt_max_level(128, w.dec_len)
    threshold = 0.5
    coeffs = pywt.wavedec(train_data[30,0,:,5], 'db18', level=maxlev)

    plt.figure()
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波
    datarec = pywt.waverec(coeffs, 'db18') 

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(128), train_data[30,0,:,5])
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("Raw signal")

    plt.subplot(2, 1, 2)
    plt.plot(range(128), datarec)
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("De-noised signal using wavelet techniques")

    plt.tight_layout()
    plt.show()