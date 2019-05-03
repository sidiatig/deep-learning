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
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class PianoGenerationModel(nn.Module):
    """Applies a multi-layer long short-term memory (LSTM) to an input
    sequence.

    Input: `(seq_len, batch)`: tensor containing the word indices
        of the input sequences.

    Output: `(batch, V, seq_len)`: tensor containing the logits for each
        sample, for each t, where V is the vocabulary size.
    """
    def __init__(self, lstm_num_hidden=32, lstm_num_layers=2):
        super(PianoGenerationModel, self).__init__()
        # Store in a dictionary arguments used to construct the model
        self.hparams = locals()
        self.hparams.pop('self')
        self.hparams.pop('__class__')

        self.lstm = nn.LSTM(input_size=3,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            batch_first=True)
        self.linear_out = nn.Linear(in_features=lstm_num_hidden,
                                    out_features=3)

        self.num_hidden = lstm_num_hidden

    def forward(self, x):
        out, _ = self.lstm(x)
        p = torch.relu(self.linear_out(out))
        return p

    def sample(self, x0, sample_length, temperature=0):
        """Generate sample sequences given initial words.

        Args
            x0: `(init_len, d)`: tensor containing the initial sequences to
                generate a mini-batch of sequences.
            sample_length (int): length of the sequence to generate
            temperature (float): 0 indicates greedy sampling, higher values
                increase randomness in the generated sequence. Default: 0
        Return
            `(batch, seq_len)` tensor containing the indices of the
            generated sequences (including the initial word).
        """
        assert sample_length > 0, "seq_length must be a positive integer"

        init_length, d = x0.shape
        seq_length = init_length + sample_length
        samples = torch.empty(seq_length, d, dtype=torch.float)
        samples[:init_length] = x0

        x = x0.unsqueeze(dim=0)
        h = torch.zeros([2, 1, self.num_hidden], dtype=torch.float)
        c = torch.zeros([2, 1, self.num_hidden], dtype=torch.float)

        with torch.no_grad():
            for i in range(init_length, seq_length):
                out, (h, c) = self.lstm(x, (h, c))
                x = torch.relu(self.linear_out(out[-1:]))
                samples[i] = x[0]

        return samples
