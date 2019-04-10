"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
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
  y_pred = np.argmax(predictions, axis=1)
  y_true = np.argmax(targets, axis=1)
  accuracy = np.sum(y_pred == y_true)/n_samples
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
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
  datasets = cifar10_utils.read_data_sets(DATA_DIR_DEFAULT)
  train_data = datasets['train']
  test_data = datasets['test']
  model = MLP(n_inputs=3072, n_hidden=dnn_hidden_units, n_classes=10)
  loss_fn = CrossEntropyModule()

  plot_every = 10
  avg_loss = 0
  losses = []
  losses_idx = []
  avg_acc = 0
  accs = []
  test_accs = []
  test_accs_idx = []
  for step in range(FLAGS.max_steps):
    x, y = train_data.next_batch(FLAGS.batch_size)
    x = x.reshape(FLAGS.batch_size, -1)
    out = model.forward(x)

    # Forward and backward passes
    loss = loss_fn.forward(out, y)
    dout = loss_fn.backward(out, y)
    model.backward(dout)

    # Parameter updates
    for layer in model.layers:
      params = getattr(layer, 'params', None)
      if params is not None:
        grads = layer.grads
        layer.params = {name: params[name] - FLAGS.learning_rate * grads[name] for name in params}

    avg_loss += loss/plot_every
    avg_acc += accuracy(out, y)/plot_every
    if (step + 1) % plot_every == 0:
      print('\r[{}/{}] train loss: {:.6f}  train acc: {:.6f}'.format(step + 1,
                                                                     FLAGS.max_steps,
                                                                     avg_loss, avg_acc), end='')
      losses.append(avg_loss)
      losses_idx.append(step)
      accs.append(avg_acc)
      avg_loss = 0
      avg_acc = 0

    # Evaluate
    if (step + 1) % FLAGS.eval_freq == 0:
      x, y = test_data.next_batch(test_data.num_examples)
      x = x.reshape(test_data.num_examples, -1)
      out = model.forward(x)
      acc = accuracy(out, y)
      print(' test accuracy: {:6f}'.format(acc))
      test_accs.append(acc)
      test_accs_idx.append(step)

  plt.plot(losses_idx, losses)
  plt.show()
  plt.plot(losses_idx, accs)
  plt.plot(test_accs_idx, test_accs)
  plt.show()
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
  train()

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