# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as DataSet
import torch.utils.data as Data

torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = True

train_data = DataSet.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist',train=False)

loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,        # 有几层 RNN layers
            batch_first=True,    # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = torch.nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
lossFunc = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):
        b_x = b_x.view(-1, 28, 28)                 # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)
        loss = lossFunc(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
