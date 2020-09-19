import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataset import Mydataset

N_CHANNEL = 20
N_FEATURE = 64
N_CLASS = 2
BATCH_SIZE = 16
N_EPOCH = 200
LR = 0.001
LR_DECAY = 1e-4

class SignalCNN(nn.Module) :
    
    def __init__(self, n_feature1, n_feature2, n_class = 2):
        super().__init__()            # input shape (1, 32, 20)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,      
                out_channels = n_feature1,    # n_filters
                kernel_size = 5,      # filter size
                stride = 1,           # filter movement/step
                padding = 2
            ), 
            # nn.BatchNorm2d(n_feature1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 16, 10)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_feature1,      
                out_channels = n_feature2,    # n_filters
                kernel_size = 3,      # filter size0
                stride = 1,           # filter movement/step
                padding = 1
            ), 
            # nn.BatchNorm2d(n_feature2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (32, 8, 5)
        )

        self.fc1 = nn.Linear(32*8*5, n_class)
        self.fc1.weight.data.normal_(0, 0.01)
        # self.fc2 = nn.Linear()
        self.batch_norm = nn.BatchNorm2d(1)
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.batch_norm(x)
        # print(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.active(x)
        return output

    def predict(self, x):
        pred = self.forward(x)
        # print(pred)
        pred = torch.argmax(pred, dim=1)
        return pred

data_dir = 'data/data5.txt'

if __name__ == "__main__":
    
    scnn = SignalCNN(16, 32, 2)
    scnn.cuda()

    optimizer = optim.Adam(scnn.parameters(), lr=LR, weight_decay=LR_DECAY)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()

    dataSet = Mydataset(data_dir)
    # testSet = Mydataset('data/sptestdata.txt')
    # test_total = len(testSet)
    total = len(dataSet)
    print(total)
    train_size = int(0.8*total)
    valid_size = total - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataSet, [train_size, valid_size])
    train_loader = DataLoader(dataset=dataSet, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(dataset=testSet, batch_size=60, shuffle=True)

    for epoch in range(N_EPOCH):
        scnn.zero_grad()
        # scheduler.step()
        for batch, (x, y, c, index) in enumerate(train_loader):
            x = Variable(x.float().cuda())
            y = Variable(y.long().squeeze().cuda())
            output = scnn(x)
        
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if((epoch+1) % 1 == 0) :
            correct = 0
            
            for batch, (x, y, c, index) in enumerate(valid_loader):
                x = Variable(x.float().cuda())
                y = Variable(y.long().squeeze().cuda())
                output = scnn.predict(x)
                # print(output, y)
                for i, o in enumerate(output):
                    if(o == y[i]): correct+=1

            #     x = x.float().cuda()
            #     y = y.long().cuda()
            #     index = index[:, 1, :]
            #     output = scnn.forward(x)
            #     if(batch % 1 == 0):
            #         print("character %d"%(batch+1))

            #     pred = torch.argmax(output, dim=1)
            #     print(pred)
            #     # test_total +=1
            #     for i, o in enumerate(pred):
            #         if(o == y[i]): correct+=1
            #     output = output[:, 1]
                
            #     print(y.cpu().numpy().reshape(1,-1))
            #     p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                

            #     for i, o in enumerate(output):
            #         p[index[i].int().item() - 1] += o.item()

            #     # print(p)
            #     i1 = p.index(max(p))
            #     offset = 0
            #     if(i1 + 1 > 6):
            #         lp = p[0:6]
            #     else :
            #         lp = p[6:12]
            #         offset = 6
            #     i2 = lp.index(max(lp))
                    
            #     print(i1+1, i2+1+offset)

            print('Epoch: %d | loss: %f | accuracy: %f'%(epoch+1, loss, correct/valid_size))

    torch.save(scnn, 'model\scnn_0_512_32fr_5.pkl')