import pandas as pd
import dataset
from torch.utils.data import DataLoader,TensorDataset
my_data = dataset.Mydataset()
my_data_loader = DataLoader(dataset=my_data,batch_size=1,shuffle=False)

for i,data in enumerate(my_data_loader):
    X,Y,char,index =data
    print(X)
    print(Y)
    print(char)
    print(index)