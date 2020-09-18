import dataset
from torch.utils.data import DataLoader,TensorDataset
my_data = dataset.Mydataset()
my_data_loader = DataLoader(dataset=my_data,batch_size=12,shuffle=False)

if __name__ == "__main__":
    # print(len(my_data))
    # print(len(my_data_loader))
    for i,data in enumerate(my_data_loader):
        X,Y,char,index =data
        print(Y.squeeze())