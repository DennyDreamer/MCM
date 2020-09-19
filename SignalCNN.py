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
N_EPOCH = 100
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
                kernel_size = 3,      # filter size
                stride = 1,           # filter movement/step
                padding = 1
            ), 
            # nn.BatchNorm2d(n_feature2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (32, 8, 5)
        )
        self.fc1 = nn.Linear(32*8*5, n_class)
        self.fc1.weight.data.normal_(0, 0.1)
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

data_dir = 'data/testdata1.txt'

if __name__ == "__main__":
    
    scnn = SignalCNN(16, 32, 2)
    scnn.cuda()

    optimizer = optim.Adam(scnn.parameters(), lr=LR, weight_decay=LR_DECAY)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()

    dataSet = Mydataset(data_dir)
    total = len(dataSet)
    train_size = int(0.8*total)
    valid_size = total - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataSet, [train_size, valid_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

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

        if((epoch+1) % 10 == 0) :
            correct = 0
            for batch, (x, y, c, index) in enumerate(valid_loader):
                x = Variable(x.float().cuda())
                y = Variable(y.long().squeeze().cuda())
                output = scnn.predict(x)
                # print(output, y)
                for i, o in enumerate(output):
                    if(o == y[i]): correct+=1

            print('Epoch: %d | loss: %f | accuracy: %f'%(epoch+1, loss, correct/valid_size))

        torch.save(scnn, 'model\model1.pkl')