import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataset import Mydataset, LSTMdataset
from SignalCNN import SignalCNN

model_dir = 'model/signal_cnn.pkl'
data_dir = 'data/testdata3.txt'

if __name__ == "__main__":
    
    model = torch.load(model_dir)
    dataset = Mydataset(data_dir)
    test_loader = DataLoader(dataset=dataset, batch_size=60, shuffle=False)

    correct = 0
    total = len(test_loader)
    for batch, (x, y, c, index) in enumerate(test_loader):
        x = x.float().cuda()
        index = index[:, 1, :]
        output = model.forward(x)
        if(batch % 1 == 0):
            print("character %d"%(batch+1))
        output = output[:, 1]
        maxIndex = torch.argmax(output, dim=0)     #第二列最大值下标
        rol = index[maxIndex]                      #最大下标所在行或列
        maxp = 0.
        lor = 0.

        for i, o in enumerate(output):
            if o > maxp and ((index[i]-6.5)*(rol-6.5) < 0):
                maxp = o
                lor = index[i]

        
        print(rol.int().item(), lor.int().item())


