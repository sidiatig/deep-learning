# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.lstm = nn.LSTM(input_size=vocabulary_size,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.linear_out = nn.Linear(in_features=lstm_num_hidden,
                                    out_features=vocabulary_size)

        one_hot_codes = torch.eye(vocabulary_size)
        self.register_buffer('one_hot_codes', one_hot_codes)

    def forward(self, x):
        x_one_hot = self.one_hot_codes[x]
        out, _ = self.lstm(x_one_hot)
        p = self.linear_out(out).transpose(1, 2)

        return p
