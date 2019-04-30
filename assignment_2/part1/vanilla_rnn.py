################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import math

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.whx = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.whh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bias_h = nn.Parameter(torch.Tensor(num_hidden))
        self.wph = nn.Parameter(torch.Tensor(num_classes, num_hidden))
        self.bias_p = nn.Parameter(torch.Tensor(num_classes))

        stdv = 1.0 / math.sqrt(num_hidden)
        for weight in [self.whx, self.whh, self.wph]:
            nn.init.uniform_(weight, -stdv, stdv)
        for weight in [self.bias_h, self.bias_p]:
            nn.init.zeros_(weight)

        self.register_buffer('h0', torch.zeros(1, num_hidden))

    def forward(self, x):
        batch_size, seq_length = x.shape

        h_prev = self.h0

        for t in range(seq_length):
            x_t = x[:, t:t+1]
            h_prev = torch.tanh(x_t.matmul(self.whx.t())
                                + h_prev.matmul(self.whh.t())
                                + self.bias_h)
        p = h_prev.matmul(self.wph.t()) + self.bias_p

        return p
