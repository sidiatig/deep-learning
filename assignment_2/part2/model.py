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
    """Applies a multi-layer long short-term memory (LSTM) to an input
    sequence.

    Input: `(seq_len, batch)`: tensor containing the word indices
        of the input sequences.

    Output: `(batch, V, seq_len)`: tensor containing the logits for each
        sample, for each t, where V is the vocabulary size.
    """
    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):
        super(TextGenerationModel, self).__init__()

        self.lstm = nn.LSTM(input_size=vocabulary_size,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers)
        self.linear_out = nn.Linear(in_features=lstm_num_hidden,
                                    out_features=vocabulary_size)

        one_hot_codes = torch.eye(vocabulary_size)
        self.register_buffer('one_hot_codes', one_hot_codes)
        self.register_buffer('init_state', torch.zeros(lstm_num_layers, 1, lstm_num_hidden))

    def forward(self, x):
        x_one_hot = self.one_hot_codes[x]
        out, _ = self.lstm(x_one_hot)
        p = self.linear_out(out).permute(1, 2, 0)

        return p

    def sample(self, x0, seq_length):
        """Applies a multi-layer long short-term memory (LSTM) to an input
        sequence.

        Input: `(batch,)`: tensor containing the initial word indices to
            generate a mini-batch of sequences.

        Output: `(batch, seq_len)`: tensor containing the indices of the
            generated sequences (including the initial word).
        """
        assert seq_length > 0, "seq_length must be a positive integer"

        batch_size, = x0.shape
        samples = torch.empty(batch_size, seq_length, dtype=torch.long)
        samples[:, 0] = x0

        x = x0.view(1, -1)
        h = self.init_state.expand(-1, batch_size, -1).contiguous()
        c = self.init_state.expand(-1, batch_size, -1).contiguous()

        for i in range(1, seq_length):
            x_one_hot = self.one_hot_codes[x]
            out, (h, c) = self.lstm(x_one_hot, (h, c))
            p = self.linear_out(out)
            _, x = p.max(dim=-1)
            samples[:, i] = x

        return samples
