<<<<<<< HEAD
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
=======
import numpy as np
import pandas as pd
tag=[0]*37
tag[0]=[]
tag[1]=[1,7]
tag[2]=[1,8]
tag[3]=[1,9]
tag[4]=[1,10]
tag[5]=[1,11]
tag[6]=[1,12]
tag[7]=[2,7]
tag[8]=[2,8]
tag[9]=[2,9]
tag[10]=[2,10]
tag[11]=[2,11]
tag[12]=[2,12]
tag[13]=[3,7]
tag[14]=[3,8]
tag[15]=[3,9]
tag[16]=[3,10]
tag[17]=[3,11]
tag[18]=[3,12]
tag[19]=[4,7]
tag[20]=[4,8]
tag[21]=[4,9]
tag[22]=[4,10]
tag[23]=[4,11]
tag[24]=[4,12]
tag[25]=[5,7]
tag[26]=[5,8]
tag[27]=[5,9]
tag[28]=[5,10]
tag[29]=[5,11]
tag[30]=[5,12]
tag[31]=[6,7]
tag[32]=[6,8]
tag[33]=[6,9]
tag[34]=[6,10]
tag[35]=[6,11]
tag[36]=[6,12]
columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

def isP300( charNum , num):
    "字母序号 行列序号"
    if num in tag[charNum]:
        return 1
    else:
        return 0


#####归一化函数#####

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))



for k in range(1,6):
    path1 = "D:/data/S"+str(k)+"/S"+str(k)+"_train_event.xlsx"
    path2 = "D:/data/S"+str(k)+"/S"+str(k)+"_train_data.xlsx"
    df = pd.DataFrame()
    for j in range(12):
        df1 = pd.read_excel(path1,sheet_name=j)
        df2 = pd.read_excel(path2,sheet_name=j)
        df2.columns = columns
        colName1 = df1.columns[0]
        colName2 = df1.columns[1]

        temp = pd.DataFrame()
        for i in range(65):
            bianhao = df1[colName1][i]
            hanghao = df1[colName2][i]
            valid = 0
            if bianhao>99:
                continue
            if isP300( colName1-100, bianhao) :
                valid = 1
                print( "true")
            bianhaoList = [bianhao]*50
            biaozhiList = [colName1]*50
            validList = [valid]*50
            unvalidList = [1-valid]*50
            df3 = pd.DataFrame({"行列号":bianhaoList,"标识":biaozhiList})
            df4 = df2.iloc[hanghao+73:hanghao+123]
            df5 = pd.DataFrame({"有效":validList,"无效":unvalidList})
            df3.index = range(50*i,50*(i+1))
            df4.index = range(50*i,50*(i+1))
            df5.index = range(50*i,50*(i+1))
            df3 = pd.concat([df3,df4,df5], axis=1)
            temp = pd.concat([temp,df3], axis=0)
        df = pd.concat([df,temp], axis=0)
    for i in range(1, 21):
        df[i] = df[[i]].apply(max_min_scaler)
    df.to_csv("data"+str(k)+".txt",index=False, sep='\t')
>>>>>>> fa1ce6ec3b07dab81e4667973958507f142786a2
