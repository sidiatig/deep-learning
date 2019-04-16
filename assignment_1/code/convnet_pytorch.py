"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized a ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=64), nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=128), nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=256), nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=256), nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=512), nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=512), nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=512), nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=512), nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.linear = nn.Linear(in_features=512, out_features=n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = x.shape[0]
    features = self.layers(x)
    out = self.linear(features.view(batch_size, -1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
