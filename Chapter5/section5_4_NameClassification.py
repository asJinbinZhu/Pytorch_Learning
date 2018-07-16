# -*- coding: utf-8 -*-

'''
构建并训练一个字符级别的循环神经网络来对单词进行分类。在基于字符级别的循环神经网络里，单词将是一个个字符（字母）的序列，利用循环神经网络
具备一定的时序记忆功能来实现判断一个单词的类别。具体在本示例中，将对来自18种语言的上千个人名进行训练，并让网络根据给定名字的字母拼写来预测
这个名字最可能是哪个国家常用的名字。
'''

# 第一步，准备数据
# downloading http://www.pytorchtutorial.com/goto/http://download.pytorch.org/tutorial/data.zip
# data/names目录下包含18个以语言名命名的txt文件，每一个文件里包含大量的人名字符串，每一行表示一个人名。大多数是常用字符，不过我们仍然需要
# 把这些人名字符串从unicode字符转化为ASCII字符

# 把所有这些数据加载到一个字典中，字典的键是语言类别字符串，对应的值是一个包含了该语言人名的列表，类似于{language:[names …]}。同时我们也
# 维护一个类别列表，保存了所有语言类别，类似于[language …]

import torch
import glob
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


def findfiles(path):
    return glob.glob(path)
# print(findfiles('data/names/*.txt'))
'''
['data/names/Irish.txt', 'data/names/Chinese.txt', 'data/names/Dutch.txt', 'data/names/Scottish.txt', 
'data/names/Russian.txt', 'data/names/French.txt', 'data/names/Korean.txt', 'data/names/Portuguese.txt', 
'data/names/Arabic.txt', 'data/names/Japanese.txt', 'data/names/English.txt', 'data/names/Vietnamese.txt', 
'data/names/Czech.txt', 'data/names/Polish.txt', 'data/names/Italian.txt', 'data/names/German.txt', 
'data/names/Spanish.txt', 'data/names/Greek.txt']
'''


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodetoascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
)
# print(unicodetoascii(u'Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
# Read a file and split into lines
import sys
reload(sys)
sys.setdefaultencoding('utf8')
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    for line in lines:
        unicodetoascii(u'' + line)
    return lines

# Build the category_lines dictionary, a list of lines per category
# category_lines字典和all_categories列表保存了我们要训练的数据


category_lines = {}
all_categories = []
for filename in findfiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines


n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def lettertoindex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def linetotensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][lettertoindex(letter)] = 1
    return tensor

#print(category_lines)
print(category_lines['Italian'][:5])

# 第二步，创建网络

# 第三步，训练网络

# 第四步，评估网络

# 第五步，自我练习
