# -*- coding: utf-8 -*-

import torch
import numpy as np

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
