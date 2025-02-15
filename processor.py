import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataset import Mydataset, LSTMdataset

N_CHANNEL = 20
N_FEATURE = 32
N_CLASS = 2
BATCH_SIZE = 100

class MyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))

class SignalProcessor(nn.Module) :

    def __init__(self, n_channel, n_feature, n_class = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size = n_channel, 
                            hidden_size = n_feature, 
                            num_layers = 1)
        self.fc = nn.Linear(in_features = n_feature, 
                            out_features = n_class)
        self.fc.weight.data.normal_(0, 0.1)
        self.normal = nn.LayerNorm((32, 32))
        self.active = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # h_0 = torch.randn(2, 32, N_FEATURE).cuda()
        # c_0 = torch.randn(2, 32, N_FEATURE).cuda()
        # out = self.e(input)
        # out = nn.BatchNorm1d(50)
        
        x, (h_n, c_n) = self.lstm(x)
        x = self.normal(x)
        x = self.fc(x[:,-1,:])
        x = self.softmax(x)
        # print(out)
        return x

    def predict(self, x):
        pred = self.forward(x)
        # print(pred)
        pred = torch.argmax(pred, dim=1)
        return pred

if __name__ == "__main__":

    n_epoch = 100
    lr = 0.01

    model = SignalProcessor(N_CHANNEL, N_FEATURE, N_CLASS)
    model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.2, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()

    # print(model)
    # train_loader = DataLoader(dataset=train_set, batch_size=12, shuffle=False, num_workers=2)
    
    dataSet = LSTMdataset()
    total = len(dataSet)
    train_size = int(0.8*total)
    valid_size = total - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataSet, [train_size, valid_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    for epoch in range(n_epoch):
        model.zero_grad()
        # scheduler.step()
        for batch, (x, y, c, index) in enumerate(train_loader):
            x = Variable(x.float().cuda())
            y = Variable(y.long().squeeze().cuda())
            output = model(x)

            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if((epoch+1) % 1 == 0) :
            correct = 0
            for batch, (x, y, c, index) in enumerate(valid_loader):
                x = Variable(x.float().cuda())
                y = Variable(y.long().squeeze().cuda())
                output = model.predict(x)
                # print(output, y)
                for i, o in enumerate(output):
                    if(y[i] != 0) and (y[i] != 1): print('error')
                    if (o == y[i]): correct+=1

            print('Epoch: %d | loss: %f | accuracy: %f'%(epoch+1, loss, correct/valid_size))

    torch.save(model, 'model\lstm_model1.pkl')