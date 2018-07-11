# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data

'''
DataLoader，用来包装数据的工具，实现有效地迭代数据。所以用户要将数据（numpy array或其他类型数据）转换成Tensor，然后再放入包装起。
'''

torch.manual_seed(1)

BATCH_SIZE = 5

# 第一步，准备Tensor数据
x = torch.linspace(1, 10, 10) # tensor([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
y = torch.linspace(10, 1, 10) # tensor([ 10.,   9.,   8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.])

# 第二步，将Tensor转换成Dataset
dataset = Data.TensorDataset(x, y)

# 第三步，将Dataset放入DataLoader
loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,           # 打乱数据
    num_workers=2,          # 多线程来读数据
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('step: ', step)
        print('batch_x: ', batch_x.numpy())
        print('batch_y: ', batch_y.numpy())

'''
('step: ', 0)
('batch_x: ', array([ 5.,  7., 10.,  3.,  4.], dtype=float32))
('batch_y: ', array([6., 4., 1., 8., 7.], dtype=float32))
('step: ', 1)
('batch_x: ', array([2., 1., 8., 9., 6.], dtype=float32))
('batch_y: ', array([ 9., 10.,  3.,  2.,  5.], dtype=float32))
('step: ', 0)
('batch_x: ', array([ 4.,  6.,  7., 10.,  8.], dtype=float32))
('batch_y: ', array([7., 5., 4., 1., 3.], dtype=float32))
('step: ', 1)
('batch_x: ', array([5., 3., 2., 1., 9.], dtype=float32))
('batch_y: ', array([ 6.,  8.,  9., 10.,  2.], dtype=float32))
('step: ', 0)
('batch_x: ', array([ 4.,  2.,  5.,  6., 10.], dtype=float32))
('batch_y: ', array([7., 9., 6., 5., 1.], dtype=float32))
('step: ', 1)
('batch_x: ', array([3., 9., 1., 8., 7.], dtype=float32))
('batch_y: ', array([ 8.,  2., 10.,  3.,  4.], dtype=float32))
'''