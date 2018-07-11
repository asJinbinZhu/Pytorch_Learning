# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

'''
训练好了一个模型, 我们当然想要保存它, 留到下次要用的时候直接提取直接用
'''

torch.manual_seed(1)

# 第一步，建立数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# 第二步，建立神经网络
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
params = reNet.state_dict()
for k, v in params.items():
    print(k, v)

# 第三步，训练网络

# optimizer 是训练的工具
optimizer = torch.optim.SGD(reNet.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
lossFun = torch.nn.MSELoss() # 预测值和真实值的误差计算公式 (均方差)

for t in range(100):
    pre = reNet(x)  # 喂给 net 训练数据 x, 输出预测值
    loss = lossFun(pre, y) # 计算两者的误差

    optimizer.zero_grad() # 清空上一步的残余更新参数值
    loss.backward() # 误差反向传播, 计算参数更新
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

# 第四步，保存模型
torch.save(reNet, 'net.pkl')  # 保存整个网络
torch.save(reNet.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
params = reNet.state_dict()
for k, v in params.items():
    print(k, v)
