import torch
import torch.nn as nn 

from torch.utils.data import Dataset, DataLoader

class Testset(Dataset):
    
    def __init__(self):
        self.data = torch.randn(1000, 50, 20)
        self.label = torch.randint(0, 2, (1000, 1, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    dataSet = Testset()
    loader = DataLoader(dataset=dataSet, batch_size=12, shuffle=False, num_workers=2)

    for batch, (x, y) in enumerate(loader):
        print(x.size())
        print(y.squeeze().size())