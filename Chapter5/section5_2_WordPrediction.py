# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable

'''
如何使用word embedding做自然语言处理的词语预测

N-Gram language Modeling:
    每一句话有很多单词组成，而对于一句话，这些单词的组成顺序也是很重要的，我们想要知道在一篇文章中我们是否可以给出几个词然后预测这些词后面
    的一个单词，比如’I lived in France for 10 years, I can speak _ .’那么我们想要做的就是预测最后这个词是French.

'''

# 第一步，数据预处理
# 给出了一段文章作为我们的训练集
CONTEXT_SIZE = 2    # 表示我们想由前面的几个单词来预测这个单词
EMBEDDING_DIM = 10  # 表示word embedding的维数
sentence = """
When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty\\'s field,
Thy youth\\'s proud livery so gazed on now,
Will be a totter\\'d weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv\\'d thy beauty\\'s use,
If thou couldst answer \\'This fair child of mine
Shall sum my count, and make my old excuse,\\'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel\\'st it cold.
""".split()

# 将单词三个分组，每个组前两个作为传入的数据，而最后一个作为预测的结果
trigram = [((sentence[i], sentence[i+1]), sentence[i+2]) for i in range(len(sentence) - 2)]

# set将重复的单词去掉
vocb = set(sentence)
word_to_idx = {word: i for i, word, in enumerate(vocb)}  # 需要给每个单词编码，也就是用数字来表示每个单词，这样才能够传入word embeding得到词向量
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}


# 第二步，创建神经网络


class NgramModel(torch.nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = torch.nn.Embedding(self.n_word, n_dim)
        self.linear1 = torch.nn.Linear(context_size * n_dim, 128)
        self.linear2 = torch.nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out, 1)

        return log_prob


ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(ngrammodel.parameters(), lr=1e-3)


# 第三步，训练模型
for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for data in trigram:
        word, label = data
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # forward
        out = ngrammodel(word)
        loss = criterion(out, label)
        running_loss += loss.data.numpy()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

# 第四步，预测
word, label = trigram[3]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = ngrammodel(word)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.data.numpy()[0]]
print('real word is {}, predict word is {}'.format(label, predict_word))
