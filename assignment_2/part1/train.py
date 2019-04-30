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

import argparse
import time
from datetime import datetime
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))

################################################################################

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    Implement accuracy computation.
    """
    n_samples = targets.shape[0]
    _, y_pred = predictions.max(dim=1)
    accuracy = (y_pred == targets).sum().item() / n_samples

    return accuracy

@ex.config
def config():
    input_length = 5
    tag = 'vanilla_rnn'
    log_train = False

@ex.capture
def train(configs, input_length, log_train, _run):

    assert configs.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(configs.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model_class = VanillaRNN
    else:
        model_class = LSTM

    model = model_class(input_length, configs.input_dim,
                        configs.num_hidden, configs.num_classes,
                        configs.batch_size, configs.device).to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(configs.input_length + 1)
    data_loader = DataLoader(dataset, configs.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), configs.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        out = model.forward(batch_inputs)
        batch_loss = criterion(out, batch_targets)
        batch_loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.max_norm)
        ############################################################################

        optimizer.step()

        loss = batch_loss.item()
        acc = accuracy(out, batch_targets)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = configs.batch_size / float(t2 - t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    configs.train_steps, configs.batch_size, examples_per_second,
                    acc, loss
            ))

            if log_train:
                _run.log_scalar('train-loss', loss, step)
                _run.log_scalar('train-acc', acc, step)

        if step == configs.train_steps:
            break

    return acc

 ################################################################################
 ################################################################################


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    # train(config)
    @ex.main
    def run_exp():
        train(config)


    @ex.command
    def seq_length_experiments():
        ex.observers.clear()
        N_EXPER = 10
        results = np.empty(N_EXPER)
        for i in range(N_EXPER):
            results[i] = train(config)

        np.save('accs_len{:d}'.format(config.input_length), results)

    ex.run(config_updates={'input_length': config.input_length,
                           'log_train': True}
          )

    # ex.run('seq_length_experiments',
    #        config_updates={'input_length': config.input_length,
    #                        'log_train': False}
    #        )

