# -*- coding: utf-8 -*-

import torch

# 第一种方法


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x)
        return x


net1 = Net(n_features=1, n_hidden=3, n_output=1)
print(net1)
'''
Net(
  (hidden): Linear(in_features=1, out_features=3, bias=True)
  (predict): Linear(in_features=3, out_features=1, bias=True)
)
'''

# 第二种方法
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1)
)
print(net2)
'''
Sequential(
  (0): Linear(in_features=1, out_features=3, bias=True)
  (1): ReLU()
  (2): Linear(in_features=3, out_features=1, bias=True)
)
'''