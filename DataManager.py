import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, numInFeature, numOutFeature):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(numInFeature, numOutFeature)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(numOutFeature, 1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return out

class myDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]

class DM(object):
    def __init__(self, path, rate, dropList):
        self.rate = rate
        self.dropList = dropList
        
        self.df = pd.read_csv(path, sep='\t')
        print('Data Loaded.')
        self.dropPartial()
        print('Households with paritial interview or proxy interview dropped.')
        self.dropAttribute()
        print('Recodes without hehelf and variables with too many -1 dropped.')
        
    def dropPartial(self):
        df = self.df[self.df['w5indout'].isin([11])]
        df = df[df['hehelf']!=-1]
        originalHousehold = self.df.groupby('idahhw5').size()
        myFilter = lambda oH : (lambda x: x.shape[0]==oH[x['idahhw5'].iat[0]])
        self.df = df.groupby('idahhw5').filter(myFilter(originalHousehold))
    
    def dropAttribute(self):
        columns = []
        for attr in self.df.columns:
            t = self.df[attr]==-1
            if t.sum()<self.rate*len(self.df):
                columns.append(attr)
        self.df=self.df[columns]
        
    def to_csv(self,name):
        self.df.to_csv(name)

if __name__ == '__main__':
    Data = DM('wave_5_elsa_data_v4.tab', 0.8, 2)
    
    df = Data.df
    df = df.loc[:,:'wpexw']
    
    dropFH = open('drop.txt', 'r')
    rawDroppedAttributeList = dropFH.readlines()
    DroppedAttributeList = [s.rstrip('\n').rstrip('\t') for s in rawDroppedAttributeList]
    DroppedAttributeSet = set(DroppedAttributeList)
    
    columns = []
    for name in df.columns:
        if name not in DroppedAttributeSet:
            columns.append(name)
            
    df = df[columns]
    
    df.to_csv('hope.csv')
    
    labels = df['hehelf']
    data = df.drop(['hehelf', 'idauniq'], axis=1)
    data[data<0] = np.NaN
    data = data.fillna(data.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    
    print('LinearRegression MSE:', sklearn.metrics.mean_squared_error(y_test, y_pred))
    
    net = Net(91, 25)
    print(net)
    
    torch_train = myDataSet(X_train.to_numpy().astype('float32'), y_train.to_numpy().astype('float32'))
    torch_test = myDataSet(X_test.to_numpy().astype('float32'), y_test.to_numpy().astype('float32'))
    trainLoader = DataLoader(torch_train, batch_size=16, shuffle=True)
    testLoader = DataLoader(torch_test, batch_size=16, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    
    for epoch in range(300):
        running_loss = 0.0
        for i,data in enumerate(trainLoader,0):
            inputs, labels = data
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch+1, running_loss/len(torch_train)))
                
    print('finished')
    