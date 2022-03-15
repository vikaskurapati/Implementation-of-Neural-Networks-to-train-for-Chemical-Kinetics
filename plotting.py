import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import os 

os.chdir("F:/Documents/Masters/HiWi/Cantera/")
# hparams = {"lr": 1e-5,
#             "batch_size": 32,
#             "neurons":10,
#             "optimizer": torch.optim.Adam,
#             "initialization": nn.init.xavier_uniform_,
#             }
# class Net(nn.Module):
#     """Neural Network Model
#     """
#     def __init__(self):
#         super(Net, self).__init__()
#         #self.model = nn.Sequential(nn.Linear(8, 10), nn.ReLU(), nn.Sigmoid(), nn.Linear(10,1), nn.Sigmoid())
#         self.model = nn.Sequential(nn.Linear(8, hparams["neurons"]), nn.BatchNorm1d(hparams["neurons"]), nn.ReLU(), nn.Linear(hparams["neurons"],1))
#         #self.model = nn.Sequential(nn.Linear(8,10), nn.ReLU(), nn.Linear(10,1))
#         self._initialize_weights()
    
#     def forward(self, x):
#         out = self.model(x)
#         return out

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 #nn.init.normal_(m.weight, 0, 0.001)
#                 hparams["initialization"](m.weight)
#                 #nn.init.constant_(m.weight, 0)
#                 nn.init.constant_(m.bias, 0)
# net = Net()
# model = torch.load("F:/Documents/Masters/HiWi/Cantera/model.pth")["net"]
# net.load_state_dict(model)

net = torch.jit.load("model.pt")

net = net.to("cpu")
net.eval()

input_nodes = np.load('input_1_1.npy')
output_nodes = np.load('output_H2_1.npy')
input_nodes[:,0] = (input_nodes[:,0] - 1884.074369553378)/433.62573988997286
input_nodes[:,1] = (input_nodes[:,1] - 0.004196851599658142)/0.005872701842874536
input_nodes[:,2] = (input_nodes[:,2] - 0.00548001627256922)/0.0044416373696016965
input_nodes[:,3] = (input_nodes[:,3] - 0.1408999623686791)/0.051084186856168334
input_nodes[:,4] = (input_nodes[:,4] - 0.00922919219697454)/0.00543497046557566
input_nodes[:,5] = (input_nodes[:,5] - 0.08393592703309831)/0.04880731005035305
input_nodes[:,6] = (input_nodes[:,6] - 0.7559036945562518)/1.1102230246251565e-16

plt.plot(output_nodes)
outp = np.zeros(output_nodes.shape)

for i,inp in enumerate(input_nodes):
    #torch.reshape(inp, (1,8))
    min_value = 0.00033168035767556386
    ptp = 0.014135837393164001
    #ptp = 1
    #min_value = 0
    outp[i] = net(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()*ptp + min_value 
    print(outp[i])

plt.plot(outp)
plt.show()