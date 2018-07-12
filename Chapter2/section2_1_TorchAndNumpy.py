# -*- coding: utf-8 -*-

import torch
import numpy as np

# 创建tensor的多种方式
tensor1 = torch.Tensor(2, 3)
print(tensor1)
'''
tensor([[-2.2565e+07,  4.5842e-41, -5.3370e+08],
        [ 4.5842e-41,  4.4842e-44,  0.0000e+00]])
'''

tensor2 = torch.rand(2, 3)
print(tensor2)
'''
tensor([[ 0.2535,  0.0926,  0.7768],
        [ 0.8807,  0.6189,  0.1471]])
'''

# Numpy 与PyTorch数据格式之间的转换
np_data = np.arange(6).reshape(2, 3)
torch_data = torch.from_numpy(np_data)
tensor2numpy = torch_data.numpy()

print 'numpy data: ', np_data
'''
numpy data:  [[0 1 2]
 [3 4 5]]
'''

print 'torch data: ', torch_data
'''
torch data:  tensor([[ 0,  1,  2],
        [ 3,  4,  5]])
'''

print'tensor to numpy', tensor2numpy
'''
tensor to numpy [[0 1 2]
 [3 4 5]]
'''

# Torch 中的数学运算
# 1 - abs 绝对值计算
data = [1, -2, 3]
tensor = torch.FloatTensor(data)
print 'numpy: ', np.abs(data)
'''
numpy:  [1 2 3]
'''

print 'tensor: ', torch.abs(tensor)
'''
tensor:  tensor([ 1.,  2.,  3.])
'''

# 2 - 三角函数 sin
print 'numpy: ', np.sin(data)
'''
numpy:  [ 0.84147098 -0.90929743  0.14112001]
'''

print 'tensor: ',  torch.sin(tensor)
'''
tensor:  tensor([ 0.8415, -0.9093,  0.1411])
'''

# 3 - mean  均值
print 'numpy: ', np.mean(data)
'''
numpy:  0.6666666666666666
'''

print 'tensor: ', torch.mean(tensor)
'''
tensor:  tensor(0.6667)
'''

# 4 - 矩阵的乘法
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)

# 正确操作
print 'numpy: ', np.matmul(data, data)
'''
numpy:  [[ 7 10]
 [15 22]]
'''

print 'pyTorch: ', torch.matmul(tensor, tensor)
'''
pyTorch:  tensor([[  7.,  10.],
        [ 15.,  22.]])
'''

# 错误的操作
# print 'numpy: ', data.dot(data)
# print 'pyTorch: ', tensor.dot(tensor)

