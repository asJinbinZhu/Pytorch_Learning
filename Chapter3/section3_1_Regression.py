# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''
神经网络是如何通过简单的形式将一群数据用一条线条来表示. 或者说, 是如何在数据当中找到他们的关系, 然后用神经网络模型来建立一个可以代表他们关系的线条.
'''

# 第一步，建立数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) # noisy y data (tensor), shape=(100, 1)

# 画图，显示数据集
#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()

# 第二步，建立神经网络
'''
直接运用 torch 中的体系. 
    先定义所有的层属性(__init__()), 
    然后再一层层搭建(forward(x))层于层的关系链接.
思考：有哪几个属性?
'''


class ReNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(ReNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        hidden_out = F.relu(self.hidden(input))
        pre_out = self.predict(hidden_out)
        return pre_out


reNet = ReNet(n_features=1, n_hidden=10, n_output=1)
# print(reNet)
'''
ReNet(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
'''

# 第三步，训练网络
'''
# optimizer 是训练的工具
optimizer = torch.optim.SGD(reNet.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
lossFun = torch.nn.MSELoss() # 预测值和真实值的误差计算公式 (均方差)

for t in range(100):
    pre = reNet(x)  # 喂给 net 训练数据 x, 输出预测值
    loss = lossFun(pre, y) # 计算两者的误差

    optimizer.zero_grad() # 清空上一步的残余更新参数值
    loss.backward() # 误差反向传播, 计算参数更新
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
'''

# 第四步，可视化训练过程
plt.ion

optimizer = torch.optim.SGD(reNet.parameters(), lr=0.2)
lossFun = torch.nn.MSELoss()

for t in range(100):
    pre = reNet(x)
    loss = lossFun(pre, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
