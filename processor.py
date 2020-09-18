import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataset import Mydataset

class SignalProcessor(nn.Module) :

    def __init__(self, n_channel, n_feature, n_class = 2):
        super().__init__()
        self.fc = nn.Linear(in_features = n_feature, 
                            out_features = n_class)
        self.lstm = nn.LSTM(input_size = n_channel, 
                            hidden_size = n_feature, 
                            num_layers = 2)

    def forward(self, input):
        out, (h_n, c_n) = self.lstm(input)
        out = self.fc(out[:,-1,:])
        out = F.log_softmax(out, dim=1)
        print(out)
        return out

    def predict(self, x):
        pred = self.forward(x)
        # pred = torch.argmax(pred, dim=1)
        return pred

if __name__ == "__main__":

    n_epoch = 10
    lr = 0.01

    model = SignalProcessor(20, 64, 2)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.NLLLoss()

    # print(model)
    # train_loader = DataLoader(dataset=train_set, batch_size=12, shuffle=False, num_workers=2)
    
    dataSet = Mydataset()
    total = len(dataSet)
    train_size = int(0.8*total)
    valid_size = total - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataSet, [train_size, valid_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=12, shuffle=False, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=12, shuffle=False, num_workers=2)

    for epoch in range(n_epoch):
        model.zero_grad()
        for batch, (x, y, c, index) in enumerate(train_loader):
            x = Variable(x.float().cuda())
            y = Variable(y.long().squeeze().cuda())
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

        # if(epoch % 2 == 1) :
        #     correct = 0
        #     for batch, (x, y, c, index) in enumerate(valid_loader):
        #         x = Variable(x.float().cuda())
        #         y = Variable(y.long().squeeze().cuda())
        #         output = model(x)
        #         print(output, y)
        #         # for i, o in enumerate(output):
        #         #     if(o == y[i]): correct+=1

        #     print('Epoch: %d | loss: %f | accuracy: %f'%(epoch, loss, correct/valid_size))

    torch.save(model, 'model1.pkl')