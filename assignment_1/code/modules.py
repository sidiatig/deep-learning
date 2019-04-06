"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

EPS = 1e-9

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.in_features = in_features
    self.out_features = out_features
    self.params = {'weight': np.random.normal(scale=0.0001, size=[out_features, in_features]),
                   'bias': np.zeros(out_features)}
    self.grads = {param: np.zeros_like(self.params[param]) for param in self.params}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Implement forward pass of the module.
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.matmul(x, self.params['weight'].T) + self.params['bias']
    self.cache = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = np.matmul(dout.T, self.cache)
    self.grads['bias'] = dout.sum(axis=0)
    dx = np.matmul(dout, self.params['weight'])
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

  def __repr__(self):
    return '({}): in_features={}, out_features={}'.format(
      self.__class__.__name__, self.in_features, self.out_features)

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Implement forward pass of the module.
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.maximum(x, 0.0)
    self.cache = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    indicator = np.where(self.cache > 0, 1.0, 0.0)
    dx = dout * indicator
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

  def __repr__(self):
    return '({})'.format(self.__class__.__name__)

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b = x.max(axis=1, keepdims=True)
    y = np.exp(x - b)
    out = y / y.sum(axis=1, keepdims=True)
    self.cache = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_outer = np.einsum('bi,bo->bio', self.cache, self.cache)
    dout_outer = np.matmul(np.expand_dims(dout, axis=1), batch_outer)
    dx = dout * self.cache - dout_outer.squeeze()
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

  def __repr__(self):
    return '({})'.format(self.__class__.__name__)

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = -np.log(x[y.astype(np.bool)] + EPS).mean()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = -y * 1/(x + EPS)
    np.divide(dx, x.shape[0], out=dx)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
