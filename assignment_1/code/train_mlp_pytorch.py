"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.model_selection import ParameterGrid

ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 4000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

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

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  n_samples = targets.shape[0]
  _, y_pred = predictions.max(dim=1)
  accuracy = (y_pred == targets).sum().item() / n_samples
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

@ex.config
def config():
  n_hidden_1 = 500
  dropout = 0.2
  lr = 1e-4
  wdecay = 1e-2

@ex.main
def train(n_hidden_1, dropout, lr, wdecay, _run):
  """
  Performs training and evaluation of MLP model. 

  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  def get_xy_tensors(batch):
    x, y = batch
    x = torch.tensor(x.reshape(-1, 3072), dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    return x, y

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  datasets = cifar10_utils.read_data_sets(DATA_DIR_DEFAULT, one_hot=False)
  train_data = datasets['train']
  test_data = datasets['test']
  model = MLP(n_inputs=3072, n_hidden=[n_hidden_1, 400], n_classes=10, dropout=dropout).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)

  log_every = 50
  avg_loss = 0
  avg_acc = 0
  for step in range(FLAGS.max_steps):
    x, y = get_xy_tensors(train_data.next_batch(FLAGS.batch_size))

    # Forward and backward passes
    optimizer.zero_grad()
    out = model.forward(x)
    loss = loss_fn(out, y)
    loss.backward()

    # Parameter updates
    optimizer.step()

    avg_loss += loss.item() / log_every
    avg_acc += accuracy(out, y) / log_every
    if step % log_every == 0:
      print('[{}/{}] train loss: {:.6f}  train acc: {:.6f}'.format(step,
                                                                     FLAGS.max_steps,
                                                                     avg_loss, avg_acc))
      _run.log_scalar('train-loss', avg_loss, step)
      _run.log_scalar('train-acc', avg_acc, step)
      avg_loss = 0
      avg_acc = 0

    # Evaluate
    if step % FLAGS.eval_freq == 0 or step == (FLAGS.max_steps - 1):
      x, y = get_xy_tensors(test_data.next_batch(test_data.num_examples))
      model.eval()
      out = model.forward(x)
      model.train()
      test_loss = loss_fn(out, y).item()
      test_acc = accuracy(out, y)
      print('[{}/{}]  test accuracy: {:6f}'.format(step, FLAGS.max_steps, test_acc))

      _run.log_scalar('test-loss', test_loss, step)
      _run.log_scalar('test-acc', test_acc, step)
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  #train()
  grid_values = {'dropout': [0.1, 0.2, 0.5],
                 'lr': [1e-3, 5e-4, 1e-4],
                 'wdecay': [1e-1, 1e-2, 1e-3],
                 'n_hidden_1': [400, 600, 1000]}
  grid = ParameterGrid(grid_values)
  for i, hparams in enumerate(grid):
   print('Experiment configuration {:d}/{:d}'.format(i + 1, len(grid)))
   print(hparams)
   ex.run(config_updates=hparams)

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
