# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:19:19 2022

@author: Vikas Kurapati
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torch_optimizer as optim

class Net(nn.Module):
    """Neural Network Model
    """
    def __init__(self):
        super(Net, self).__init__()
        #self.model = nn.Sequential(nn.Linear(8, 10), nn.BatchNorm1d(10), nn.ReLU(), nn.Linear(10,1))
        #self.model = nn.Sequential(nn.Linear(7,10), nn.BatchNorm1d(10), nn.ReLU(),nn.Linear(10,1), nn.Sigmoid())
        self.model = nn.Sequential(nn.Linear(7,10), nn.ReLU(), nn.Linear(10,1))
        #self.model = nn.Sequential(nn.Linear(7,10), nn.ReLU(), nn.Linear(10,1), nn.Sigmoid())
        #self.model = nn.Sequential(nn.Linear(1,10), nn.ReLU(), nn.Linear(10,1), nn.Sigmoid())
        #self.model = nn.Sequential(nn.Linear(8,10),nn.ReLU(),nn.Linear(10,10),nn.Sigmoid(),nn.Linear(10, 1))
        #self.model = nn.Sequential(nn.Linear(8,10), nn.ReLU(), nn.Linear(10,1))
        self._initialize_weights()
    
    def forward(self, x):
        out = self.model(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, 0, 0.001)
                nn.init.xavier_uniform_(m.weight)
                #nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                #nn.init.normal_(m.bias, 0, 0.001)

class Nodes(Dataset):
    """TensorDataset with support of transforms.
    """
    #seed = 42
    def __init__(self,train=True, transform=None):
        self.tensors= [torch.Tensor(np.load('input_standardized_1.npy')),torch.Tensor(np.load('output_O2_1.npy'))]

    def __getitem__(self, index):
        x = self.tensors[0][index]

        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

net = Net()
criterion = nn.MSELoss(reduction='sum')
dataset = Nodes()

train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)



#optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.7, weight_decay=1e-5)
#optimizer = optim.Adahessian(net.parameters(),lr= 1.0,betas= (0.9, 0.999),eps= 1e-4,weight_decay=0.0,hessian_power=1.0)

num_epochs = 10000

#num_epochs = 50000

train_loss = np.zeros(num_epochs)
test_loss= np.zeros(num_epochs)
loss_ = 10

#net = net.to("cuda:0")
for epoch in range(num_epochs):
    #net = net.to("cuda:0")
    net.train()
    running_loss = 0
    for batch_id, (inputs, targets) in enumerate(train_dataloader):
        #inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = torch.reshape(targets, outputs.shape)
        #print(outputs.shape, targets.shape)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        #print(loss)
        #running_loss += loss
        loss.backward()
        optimizer.step()
    train_loss[epoch] = running_loss/len(train_dataloader)
    
    net.eval()
    running_loss = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(test_dataloader):
            #inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
            outputs = net(inputs)
            targets = torch.reshape(targets, outputs.shape)
            running_loss += criterion(outputs, targets).item()
    test_loss[epoch] = running_loss/len(test_dataloader)
        
    if (0.5*(train_loss[epoch]+test_loss[epoch]) < loss_):
        loss_ = 0.5*(train_loss[epoch]+test_loss[epoch])
        model_scripted = torch.jit.script(net)
        model_scripted.save("model.pt")
    if (epoch%100 == 0):
        print("Epoch: " + str(epoch))
        print("Loss" + " : " + str(0.5*(train_loss[epoch]+test_loss[epoch])))  
        
plt.plot(train_loss, label='Training')
plt.plot(test_loss, label='Test')
plt.legend()
plt.show()
'''
model_scripted = torch.jit.script(net)
model_scripted.save('model.pt')


plt.plot(loss_array, '*')
plt.show()

input_nodes = np.load('input_1_1.npy')
output_nodes = np.load('output_H2_1.npy')
plt.semilogy(output_nodes)
outp = np.zeros_like(output_nodes)

for i,inp in enumerate(input_nodes):
    outp[i] = net(torch.Tensor(inp)).item()
plt.semilogy(outp, "*")
plt.show()
'''
