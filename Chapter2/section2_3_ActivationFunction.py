# -*- coding: utf-8 -*-

'''
Q1, 为什么要使用激励函数?
A1, 为了解决我们日常生活中 不能用线性方程所概括的问题.
'''

'''
Q2, 如何在神经网络中达成我们描述非线性的任务
A2, 把整个网络简化成这样一个式子. Y = Wx, W 就是我们要求的参数, y 是预测值, x 是输入值. 用这个式子, 我们很容易就能描述刚刚的那个线性问题, 因为 W 求出来可以是一个固定的数. 不过这似乎并不能让这条直线变得扭起来 , 激励函数见状, 拔刀相助, 站出来说道: “让我来掰弯它!”.
'''

'''
Q3, 激励函数
A3, Y = AF(Wx). AF 就是指的激励函数. 激励函数拿出自己最擅长的”掰弯利器”, 套在了原函数上 用力一扭, 原来的 Wx 结果就被扭弯了. 常用的激励函数有：relu, sigmoid, tanh.
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# data
x = torch.linspace(-5, 5, 200) # x data (tensor), shape=(100, 1)
x = Variable(x, requires_grad=True)

# 套用激励函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

# 画图
import matplotlib.pyplot as plt
x_np = x.data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
