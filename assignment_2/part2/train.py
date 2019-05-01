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
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

from sacred import Experiment
from sacred.observers import MongoObserver

###############################################################################
ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))

SAVE_PATH = './saved/'


def eval_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      targets: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    Implement accuracy computation.
    """
    batch_size, seq_length = targets.shape
    n_predictions = batch_size * seq_length
    _, y_pred = predictions.max(dim=1)
    accuracy = (y_pred == targets).sum().item() / n_predictions

    return accuracy


@ex.command
def train(_run):
    config = argparse.Namespace(**_run.config)

    # Initialize the device
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    total_samples = int(config.train_steps*config.batch_size)
    sampler = RandomSampler(dataset, replacement=True, num_samples=total_samples)
    data_sampler = BatchSampler(sampler, config.batch_size, drop_last=False)
    data_loader = DataLoader(dataset, num_workers=1, batch_sampler=data_sampler)

    # Initialize the model that we are going to use
    model = TextGenerationModel(dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers).to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        batch_inputs = torch.stack(batch_inputs).to(device)
        batch_targets = torch.stack(batch_targets).t().to(device)

        optimizer.zero_grad()
        logits = model(batch_inputs)
        batch_loss = criterion(logits, batch_targets)
        batch_loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            accuracy = eval_accuracy(logits, batch_targets)
            loss = batch_loss.item()
            log_str = ("[{}] Train Step {:04d}/{:04d}, "
                       "Batch Size = {}, Examples/Sec = {:.2f}, "
                       "Accuracy = {:.2f}, Loss = {:.3f}")
            print(log_str.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                 step, config.train_steps, config.batch_size,
                                 examples_per_second, accuracy, loss))

            _run.log_scalar('loss', loss, step)
            _run.log_scalar('acc', accuracy, step)

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            print('-' * config.sample_length)
            x0 = torch.randint(low=0, high=dataset.vocab_size, size=(4, 5))
            samples = model.sample(x0, config.sample_length).detach().cpu()
            samples = samples.numpy()

            for sample in samples:
                print(dataset.convert_to_string(sample))

            print('-' * config.sample_length)

        if step == config.train_steps:
            break

    print('Done training.')
    ckpt_path = os.path.join(SAVE_PATH, str(config.timestamp) + '.pt')
    torch.save({'state_dict': model.state_dict(),
                'hparams': model.hparams,
                'ix_to_char': dataset.ix_to_char},
               ckpt_path)
    print('Saved checkpoint to {}'.format(ckpt_path))


@ex.command
def generate(_run):
    config = argparse.Namespace(**_run.config)

    # Load model and vobulary mappings
    checkpoint = torch.load(config.model_path, map_location='cpu')
    model = TextGenerationModel(**checkpoint['hparams'])
    model.load_state_dict(checkpoint['state_dict'])
    ix_to_char = checkpoint['ix_to_char']
    char_to_ix = {v: k for k, v in ix_to_char.items()}

    # Prepare initial sequence
    x0 = torch.tensor([char_to_ix[char] for char in config.initial_seq],
                      dtype=torch.long).view(-1, 1)

    # Generate
    samples = model.sample(x0, config.sample_length).detach().cpu().squeeze()

    text = ''.join(ix_to_char[ix.item()] for ix in samples)
    print(text)

###############################################################################
###############################################################################


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('command', type=str, choices=['train', 'generate'],
                        default='train')

    # Model params
    parser.add_argument('--txt_file', type=str, default=None,
                        help="Path to a .txt file for training")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=100000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--sample_length', type=int, default=100,
                        help='Length of test sample sequences')

    # Text generation parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path of the saved model for generation')
    parser.add_argument('--initial_seq', type=str, default=' ',
                        help='Sequence to initialize generation')

    args = parser.parse_args()

    # noinspection PyUnusedLocal
    @ex.config
    def config_init():
        txt_file = args.txt_file
        seq_length = args.seq_length
        lstm_num_hidden = args.lstm_num_hidden
        lstm_num_layers = args.lstm_num_layers
        device = args.device
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        learning_rate_decay = args.learning_rate_decay
        learning_rate_step = args.learning_rate_step
        dropout_keep_prob = args.dropout_keep_prob
        train_steps = args.train_steps
        max_norm = args.max_norm
        summary_path = args.summary_path
        print_every = args.print_every
        sample_every = args.sample_every
        sample_length = args.sample_length
        timestamp = int(datetime.now().timestamp())
        model_path = args.model_path
        initial_seq = args.initial_seq

    if args.command == 'generate':
        ex.observers.clear()

    ex.run(args.command)
