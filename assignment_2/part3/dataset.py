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

import os
import numpy as np
import torch.utils.data as data

from mido import MidiFile


class MidiDataset(data.Dataset):

    def __init__(self, filename, seq_length):
        assert os.path.splitext(filename)[1] == ".mid"
        self._seq_length = seq_length

        file = MidiFile(filename)

        # Search for track with most notes
        track_idx = 0
        max_notes = 0
        for i, track in enumerate(file.tracks):
            if len(track) > max_notes:
                max_notes = len(track)
                track_idx = i

        track = file.tracks[track_idx]
        meta = []
        note_msgs = []

        for msg in track:
            if msg.is_meta:
                meta.append(msg)
                continue
            if msg.type == 'note_on':
                note_msgs.append(msg)

        self._data = np.empty([len(note_msgs), 3], dtype=np.float32)
        for i, msg in enumerate(note_msgs):
            self._data[i, 0] = msg.note
            self._data[i, 1] = msg.velocity
            self._data[i, 2] = msg.time

        self._data_size = self._data.shape[0]
        print("Read dataset with {:d} notes".format(self._data_size))

    def __getitem__(self, item):
        offset = np.random.randint(0, len(self._data)-self._seq_length-2)
        inputs = self._data[offset:offset+self._seq_length]
        targets = self._data[offset+1:offset+self._seq_length+1]
        return inputs, targets

    def __len__(self):
        return self._data_size
