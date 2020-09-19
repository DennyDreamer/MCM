import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataset import Mydataset, LSTMdataset

model_dir = 'model\signal_cnn.pkl'

if __name__ == "__main__":
    
    model = torch.load(model_dir)
    dataset = Mydataset("data\data1_32_norm.txt")
    