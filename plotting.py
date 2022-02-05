import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import os 

#os.chdir("F:/Documents/Masters/HiWi/Cantera/")
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

net = torch.jit.load('model.pt')

net = net.to("cpu")
net.eval()

input_nodes = np.load('input_1_1.npy')
output_nodes = np.load('output_H2_1.npy')
plt.semilogy(output_nodes, '+')
outp = np.zeros(output_nodes.shape)

for i,inp in enumerate(input_nodes):
    #torch.reshape(inp, (1,8))
    outp[i] = net(torch.reshape(torch.Tensor(inp),(1,8))).detach().numpy()
    print(outp[i])

plt.semilogy(outp, "*")
plt.show()