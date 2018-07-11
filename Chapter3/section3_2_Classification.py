# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(1)

'''
神经网络是怎么进行事物的分类
'''

# 第一步，建立数据集
n_data = torch.ones(100, 2)         # data (tensor), shape=(100, 2)
x0 = torch.normal(2*n_data, 1)      # data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # data (tensor), shape=(100, 1)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

# 画图，显示数据集
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 第二步，建立神经网络


class ClNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(ClNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


clNet = ClNet(n_features=2, n_hidden=10, n_output=2)
# print(clNet)
'''
ClNet(
  (hidden): Linear(in_features=2, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=2, bias=True)
)
'''

# 第三步，训练网络，同时可视化训练过程
optimizer = torch.optim.SGD(clNet.parameters(), lr=0.02)
lossFunc = torch.nn.CrossEntropyLoss()

# 可视化
plt.ion()

for t in range(100):
    out = clNet(x)                 # input x and predict based on x
    loss = lossFunc(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()