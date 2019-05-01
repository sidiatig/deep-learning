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
    def __init__(self, vocab_size, lstm_num_hidden=256, lstm_num_layers=2):
        super(TextGenerationModel, self).__init__()
        # Store in a dictionary arguments used to construct the model
        self.hparams = locals()
        self.hparams.pop('self')
        self.hparams.pop('__class__')

        self.lstm = nn.LSTM(input_size=vocab_size,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers)
        self.linear_out = nn.Linear(in_features=lstm_num_hidden,
                                    out_features=vocab_size)

        one_hot_codes = torch.eye(vocab_size)
        self.register_buffer('one_hot_codes', one_hot_codes)
        init_state = torch.zeros(lstm_num_layers, 1, lstm_num_hidden)
        self.register_buffer('init_state', init_state)

    def forward(self, x):
        x_one_hot = self.one_hot_codes[x]
        out, _ = self.lstm(x_one_hot)
        p = self.linear_out(out).permute(1, 2, 0)

        return p

    def sample(self, x0, sample_length):
        """Generate sample sequences given initial words.

        Args
            x0: `(init_len, batch)`: tensor containing the initial sequences to
                generate a mini-batch of sequences.
            sample_length (int): length of the sequence to generate

        Return
            `(batch, seq_len)` tensor containing the indices of the
            generated sequences (including the initial word).
        """
        assert sample_length > 0, "seq_length must be a positive integer"

        init_length, batch_size = x0.shape
        seq_length = init_length + sample_length
        samples = torch.empty(seq_length, batch_size,
                              dtype=torch.long)
        samples[:init_length] = x0

        x = x0
        h = self.init_state.expand(-1, batch_size, -1).contiguous()
        c = self.init_state.expand(-1, batch_size, -1).contiguous()

        with torch.no_grad():
            for i in range(init_length, seq_length):
                x_one_hot = self.one_hot_codes[x]
                _, (h, c) = self.lstm(x_one_hot, (h, c))
                h_out = h[-2:-1]

                p = self.linear_out(h_out)
                _, x = p.max(dim=-1)
                samples[i] = x

        return samples.t()
