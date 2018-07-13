# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F

'''
定义好一个LSTM网络，然后给出一个句子，每个句子都有很多个词构成，每个词可以用一个词向量表示，这样一句话就可以形成一个序列，
我们将这个序列依次传入LSTM，然后就可以得到与序列等长的输出，每个输出都表示的是一种词性，比如名词，动词之类的，还是一种分类问题，
每个单词都属于几种词性中的一种。

引入字符来增强表达，什么意思呢？也就是说一个单词有一些前缀和后缀，比如-ly这种后缀很大可能是一个副词，这样我们就能够在字符水平得到一个词性
判断的更好结果。具体怎么做呢？还是用LSTM。每个单词有不同的字母组成，比如 apple 由a p p l e构成，我们同样给这些字符词向量，这样形成了一个
长度为5的序列，然后传入另外一个LSTM网络，只取最后输出的状态层作为它的一种字符表达，我们并不需要关心到底提取出来的字符表达是什么样的，
在learning的过程中这些都是会被更新的参数，使得最终我们能够正确预测。
'''

# 第一步，数据预处理
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

alphabet = 'abcdefghijklmnopqrstuvwxyz'


# 第1.1 步，创建字典
word_to_idx = {}
tag_to_idx = {}
character_to_idx = {}

for context, tag in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)

for i in range(len(alphabet)):
    character_to_idx[alphabet[i]] = i

# print(word_to_idx)
# print(tag_to_idx)
# print(character_to_idx)
'''
{'Everybody': 5, 'ate': 2, 'apple': 4, 'that': 7, 'read': 6, 'dog': 1, 'book': 8, 'the': 3, 'The': 0}
{'DET': 0, 'NN': 1, 'V': 2}
{'a': 0, 'c': 2, 'b': 1, 'e': 4, 'd': 3, 'g': 6, 'f': 5, 'i': 8, 'h': 7, 'k': 10, 'j': 9, 'm': 12, 'l': 11, 'o': 14, 
'n': 13, 'q': 16, 'p': 15, 's': 18, 'r': 17, 'u': 20, 't': 19, 'w': 22, 'v': 21, 'y': 24, 'x': 23, 'z': 25}
'''


class CharLSTM(torch.nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()

        self.char_embedding = torch.nn.Embedding(n_char, char_dim)
        self.char_lstm = torch.nn.LSTM(char_dim, char_hidden, batch_first=True)

    def forward(self, x):
        x = self.char_embedding(x)  # 传入n个字符, 然后通过nn.Embedding得到词向量
        _, h = self.char_lstm(x)    # 传入LSTM网络，得到状态输出h，然后通过h[1]得到我们想要的hidden state
        return h[1]


class LSTMTagger(torch.nn.Module):
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden, n_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embedding = torch.nn.Embedding(n_word, n_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = torch.nn.LSTM(n_dim + char_hidden, n_hidden, batch_first=True)
        self.linear1 = torch.nn.Linear(n_hidden, n_tag)

    def forward(self, x, word_data):
        word = [i for i in word_data]
        char = torch.FloatTensor()
        for each in word:
            word_list = []
            for letter in each:
                word_list.append(character_to_idx[letter.lower()])
            word_list = torch.LongTensor(word_list)
            word_list = word_list.unsqueeze(0)
            tempchar = self.char_lstm(Variable(word_list))
            tempchar = tempchar.squeeze(0)
            char = torch.cat((char, tempchar.cpu().data), 0)
        char = char.squeeze(1)
        char = Variable(char)
        x = self.word_embedding(x)
        x = torch.cat((x, char), 1)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.linear1(x)
        y = F.log_softmax(x)
        return y


'''
n_word 和 n_dim来定义单词的词向量维度，n_char和char_dim来定义字符的词向量维度，char_hidden表示CharLSTM输出的维度，n_hidden表示
每个单词作为序列输入的LSTM输出维度，最后n_tag表示输出的词性的种类。

接着开始前向传播，不仅要传入一个编码之后的句子，同时还需要传入原本的单词，因为需要对字符做一个LSTM，所以传入的参数多了一个word_data表示
一个句子的所有单词。然后就是将每个单词传入CharLSTM，得到的结果和单词的词向量拼在一起形成一个新的输入，将输入传入LSTM里面，得到输出，最后
接一个全连接层，将输出维数定义为label的数目。
'''

model = LSTMTagger(
    len(word_to_idx), len(character_to_idx), 10, 100, 50, 128, len(tag_to_idx))
if torch.cuda.is_available():
    model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


def make_sequence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx


for epoch in range(300):
    print('*' * 10)
    print('epoch {}'.format(epoch + 1))
    running_loss = 0
    for data in training_data:
        word, tag = data
        word_list = make_sequence(word, word_to_idx)
        tag = make_sequence(tag, tag_to_idx)
        if torch.cuda.is_available():
            word_list = word_list.cuda()
            tag = tag.cuda()
        # forward
        out = model(word_list, word)
        loss = criterion(out, tag)
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {}'.format(running_loss / len(data)))
print()
input = make_sequence("Everybody ate the apple".split(), word_to_idx)
if torch.cuda.is_available():
    input = input.cuda()


out = model(input, "Everybody ate the apple".split())
print(out)
'''
tensor([[-2.9232, -0.1479, -2.4800],
        [-2.2661, -1.6937, -0.3390],
        [-0.2538, -1.9511, -2.5004],
        [-2.6485, -0.1100, -3.3990]])
一共有4行，每行里面取最大的，那么第一个词的词性就是NN，第二个词是V，第三个词是DET，第四个词是NN。
'''


