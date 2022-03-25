# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:21:10 2022

@author: Vikas Kurapati
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import os 

os.chdir("F:/Documents/Masters/HiWi/Cantera/")

net = torch.jit.load("model.pt")

net = net.to("cpu")
net.eval()

input_nodes = np.load('input_1_1.npy')
output_nodes = np.load('output_O_1.npy')
input_nodes[:,0] = (input_nodes[:,0] - 1884.074369553378)/433.62573988997286
input_nodes[:,0] = -1 + 2.0*(input_nodes[:,0] + 1.577568693177102)/2.4984734392012085
input_nodes[:,1] = (input_nodes[:,1] - 0.004196851599658142)/0.005872701842874536
input_nodes[:,1] = -1 + 2.0*(input_nodes[:,1] + 0.6578026261588739)/2.4066852553305984
input_nodes[:,2] = (input_nodes[:,2] - 0.00548001627256922)/0.0044416373696016965
input_nodes[:,2] = -1 + 2.0*(input_nodes[:,2] + 1.2337829085449719)/3.921584082993612
input_nodes[:,3] = (input_nodes[:,3] - 0.1408999623686791)/0.051084186856168334
input_nodes[:,3] = -1 + 2.0*(input_nodes[:,3] + 0.623821397036226)/2.3607351211584477
input_nodes[:,4] = (input_nodes[:,4] - 0.00922919219697454)/0.00543497046557566
input_nodes[:,4] = -1 + 2.0*(input_nodes[:,4] + 1.6981126678481413)/2.5963241667141967
input_nodes[:,5] = (input_nodes[:,5] - 0.08393592703309831)/0.04880731005035305
input_nodes[:,5] = -1 + 2.0*(input_nodes[:,5] + 1.7197408942739134)/2.454892711296279
input_nodes[:,6] = (input_nodes[:,6] - 0.7559036945562518)/1.1102230246251565e-16
input_nodes[:,6] = -1 + 2.0*(input_nodes[:,6] - 1.0)/(2.220446049250313e-16)

plt.plot(output_nodes)
outp = np.zeros(output_nodes.shape)

for i,inp in enumerate(input_nodes):
    #torch.reshape(inp, (1,8))
    mean = 0.005499995502600784
    std = 0.004424548471831682
    #ptp = 1
    #min_value = 0
    min_ = -1.2430636631665246
    ptp = 3.936730369347936
    outp[i] = (net(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()+1.0)*ptp*0.5 + min_
    outp[i] = outp[i]*std + mean
    #outp[i] = net(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()
    print(outp[i])

plt.plot(outp)
plt.show()