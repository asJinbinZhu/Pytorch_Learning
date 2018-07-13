# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

'''
对于分类问题，我们会使用one-hot编码，比如一共有5类，那么属于第二类的话，它的编码就是(0, 1, 0, 0, 0)。
对于单词，这样做就不行了，比如有1000个不同的词，那么使用one-hot这样的方法效率就很低了，所以我们必须要使用另外一种方式去定义每一个单词，
这就引出了word embedding。
'''

'''
在pyTorch里面word embedding实现是通过一个函数来实现的nn.Embedding
'''
# 第一步，创建字典
word_to_ix = {'hello': 0, 'world': 1}  # 每个单词我们需要用一个数字去表示他，这样我们需要hello的时候，就用0来表示它

# 第二步，设置词向量格式
embds = torch.nn.Embedding(2, 5)    # 2表示有2个词，5表示5维，其实也就是一个2×5的矩阵，
                                    # 有1000个词，每个词希望是100维，可以建立一个word embedding，nn.Embedding(1000, 100)

# 第三步，获取词向量（读取字典）
# 访问词的词向量
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
print(hello_idx)
'''
tensor([ 0])
'''
# # 第三步，获取词向量 (字典向量化)
hello_embed = embds(hello_idx)  # 得到word embedding里面关于hello这个词的初始词向量
print(hello_embed)
'''
tensor([[ 1.0362, -0.1632,  1.2962,  0.7671,  0.0224]])
'''

