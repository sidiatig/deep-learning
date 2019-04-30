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
from torch.nn import Parameter
from torch import tanh, sigmoid

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.wg = Parameter(torch.Tensor(num_hidden, input_dim + num_hidden))
        self.bg = Parameter(torch.Tensor(num_hidden))

        self.wi = Parameter(torch.Tensor(num_hidden, input_dim+num_hidden))
        self.bi = Parameter(torch.Tensor(num_hidden))

        self.wf = Parameter(torch.Tensor(num_hidden, input_dim + num_hidden))
        self.bf = Parameter(torch.Tensor(num_hidden))

        self.wo = Parameter(torch.Tensor(num_hidden, input_dim + num_hidden))
        self.bo = Parameter(torch.Tensor(num_hidden))

        self.wp = Parameter(torch.Tensor(num_classes, num_hidden))
        self.bp = Parameter(torch.Tensor(num_classes))

        for weight in [self.wg, self.wi, self.wf, self.wo, self.wp]:
            nn.init.xavier_uniform_(weight)

        for bias in [self.bg, self.bi, self.bo, self.bp]:
            nn.init.zeros_(bias)

        nn.init.ones_(self.bf)

        self.register_buffer('h0', torch.zeros(1, num_hidden))
        self.register_buffer('c0', torch.zeros(1, num_hidden))

    def forward(self, x):
        batch_size, seq_length = x.shape

        h_prev = self.h0.expand(batch_size, -1)
        c_prev = self.c0

        for t in range(seq_length):
            x_t = x[:, t:t+1]
            x_h = torch.cat((x_t, h_prev), dim=-1)

            g = tanh(x_h.matmul(self.wg.t()) + self.bg)
            i = sigmoid(x_h.matmul(self.wi.t()) + self.bi)
            f = sigmoid(x_h.matmul(self.wf.t()) + self.bf)
            o = sigmoid(x_h.matmul(self.wo.t()) + self.bo)

            c = g * i + c_prev * f
            h_prev = tanh(c) * o
            c_prev = c

        p = h_prev.matmul(self.wp.t()) + self.bp

        return p
