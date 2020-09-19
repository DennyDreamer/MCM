import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import sleep_dataset
import numpy

from torchvision import models 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

N_CHANNEL = 20
N_FEATURE = 64
N_CLASS = 5
BATCH_SIZE = 32
N_EPOCH = 100
LR = 0.001
LR_DECAY = 1e-4

class SignalFC(nn.Module) :

    def __init__(self, n_feature1, n_feature2, n_classes):

        super().__init__() 
        self.fc1 = nn.Sequential(
            nn.Linear(4, n_feature1),
            # nn.BatchNorm1d(n_feature1),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc2 = nn.Linear(n_feature1, n_classes)
        self.batch_norm = nn.BatchNorm1d(4)
        self.classificator = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x = self.batch_norm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = F.dropout(x, p = 0.1)
        # x = F.relu(x)
        # x = self.fc3(x)
        x = self.classificator(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        pred = torch.argmax(x, dim=1)
        _, ins = torch.topk(x, 2)
        return pred, ins

if __name__ == "__main__":
    
    sfc = SignalFC(16, 64, 5) 
    sfc = sfc.cuda()

    optimizer = optim.Adam(sfc.parameters(), lr=LR, weight_decay=LR_DECAY)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 33, gamma = 0.1, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()

    dataSet = sleep_dataset.sleepDataset()
    total = len(dataSet)
    train_size = int(0.7*total)
    valid_size = total - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataSet, [train_size, valid_size])
    train_loader = DataLoader(dataset=dataSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for epoch in range(N_EPOCH):
        sfc.zero_grad()
        # scheduler.step()
        for batch, (x, y) in enumerate(train_loader):
            x = Variable(x.float().cuda())
            y = Variable(y.long().squeeze().cuda())

            output = sfc(x)
        
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            correct = 0
            correct2 = 0
            dic = numpy.zeros((5, 5))
            for batch, (x, y) in enumerate(valid_loader):
                x = Variable(x.float().cuda())
                y = Variable(y.long().squeeze())
                pred, ins = sfc.predict(x)
                pred = pred.cpu()
                ins = ins.cpu()
                for i, o in enumerate(pred):
                    if o == y[i]: correct+=1
                    else : dic[y[i]][o] += 1
                    if (y.numpy()[i] in ins.numpy()) == True: correct2 += 1

            # print(dic)
            print('Epoch: %d | loss: %f | top1: %f, top2: %f'%(epoch+1, loss, correct/valid_size, correct2/valid_size))

        torch.save(sfc, 'model1.pkl')