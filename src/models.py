# src/models.py

import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from torchshape import tensorshape

class SNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=1024, num_steps=25, beta=0.95):
        super(SNN, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        self.num_steps = num_steps

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1, spk2 = None, None
        spk_rec = []
        for step in range(self.num_steps):
            cur1 = self.linear1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.linear2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
        return torch.stack(spk_rec), mem2

class CNN(nn.Module):
    def __init__(self, input_shape, num_outputs, num_hidden=1024):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        outshape = [*tensorshape(self.conv2, tensorshape(self.conv1, input_shape))][1:]
        self.linear1 = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor(outshape)), num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        return F.log_softmax(x, dim=1)