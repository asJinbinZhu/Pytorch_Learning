# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

'''
Q1, 如何定义，使用变量
'''
# 第一步，定义tensor
tensor = torch.FloatTensor([[1, 2], [3, 4]])

# 第二步，将tensor放入Variable
variable = Variable(tensor, requires_grad=True)

# 第三步，使用变量
print tensor
'''
tensor([[ 1.,  2.],
        [ 3.,  4.]])
'''

print variable
'''
tensor([[ 1.,  2.],
        [ 3.,  4.]])
'''

'''
Q2, Variable 计算, 梯度
'''
t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)
print(t_out)
'''
tensor(7.5000)
'''

print(v_out)
'''
tensor(7.5000)
'''

# 模拟 v_out 的误差反向传递
v_out.backward()
print(variable.grad)
'''
tensor([[ 0.5000,  1.0000],
        [ 1.5000,  2.0000]])
'''

'''
Q3, 获取 Variable 里面的数据(直接print(variable)只会输出 Variable 形式的数据, 在很多时候是用不了的(比如想要用 plt 画图), 所以我们要转换一下, 将它变成 tensor 形式.)
'''
print(variable) # Variable 形式
'''
tensor([[ 1.,  2.],
        [ 3.,  4.]])
'''

print(variable.data) # tensor 形式
'''
tensor([[ 1.,  2.],
        [ 3.,  4.]])
'''

print(variable.data.numpy()) # numpy 形式
'''
[[1. 2.]
 [3. 4.]]
'''

