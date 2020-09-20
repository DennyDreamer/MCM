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
data_dir = 'data/testdata1.txt'
0
if __name__ == "__main__":
    
    model = torch.load(model_dir)
    dataset = Mydataset(data_dir)
    test_loader = DataLoader(dataset=dataset, batch_size=60, shuffle=False)

    correct = 0
    total = len(dataset)
    for batch, (x, y, c, index) in enumerate(test_loader):
        x = x.float().cuda()
        y = y.long().cuda()
        index = index[:, 1, :]
        output = model.forward(x)
        if(batch % 1 == 0):
            print("character %d"%(batch+1))

        pred = torch.argmax(output, dim=1)
        for i, o in enumerate(pred):
            if(o == y[i]): correct+=1
        output = output[:, 1]
        
        # print(y)
        p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        

        for i, o in enumerate(output):
            p[index[i].int().item() - 1] += o.item()

        # print(p)
        i1 = p.index(max(p))
        offset = 0
        if(i1 + 1 > 6):
            lp = p[0:6]
        else :
            lp = p[6:12]
            offset = 6
        i2 = lp.index(max(lp))
            
        print(i1+1, i2+1+offset)
    print("accur: %f"%(correct/total))



