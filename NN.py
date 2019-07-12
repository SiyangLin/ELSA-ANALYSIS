
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
    
def train(net, X_train, y_train):
    torchTrainSet = myDataSet(X_train, y_train)
    trainLoader=DataLoader(torchTrainSet, batch_size=16, shuffle=True)
    
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
        print('[%d] loss: %.3f' % (epoch+1, running_loss/len(torchTrainSet)))
                
    print('finished')

    