# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

# 第一步，建立神经网络


class ReNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(ReNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        hidden_out = F.relu(self.hidden(input))
        pre_out = self.predict(hidden_out)
        return pre_out


reNet = ReNet(n_features=1, n_hidden=3, n_output=1)

# 第二步，读取网络参数
reNet.load_state_dict(torch.load('net_params.pkl'))

params = reNet.state_dict()
for k, v in params.items():
    print(k, v)