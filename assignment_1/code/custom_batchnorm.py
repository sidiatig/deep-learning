import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_neurons = n_neurons
    self.gamma = nn.Parameter(torch.Tensor(n_neurons))
    self.beta = nn.Parameter(torch.Tensor(n_neurons))
    nn.init.ones_(self.gamma)
    nn.init.zeros_(self.beta)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    if input.dim() != 2:
      raise ValueError('Expected 2D tensor but got {:d}D'.format(input.dim()))
    if input.shape[1] != self.n_neurons:
      msg = 'Number of neurons do not match, expected {:d}, got {:d}'
      raise ValueError(msg.format(self.n_neurons, input.shape[1]))

    mean = input.mean(dim=0)
    var = torch.mean((input - mean)**2, dim=0)
    normalized = (input - mean)/torch.sqrt(var + 1e-9)
    out = self.gamma * normalized + self.beta
    ########################
    # END OF YOUR CODE    #
    #######################

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shape (n_neurons)
      beta: mean bias tensor, applied per neuron, shape (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    mean = input.mean(dim=0)
    var = torch.mean((input - mean) ** 2, dim=0)
    sqrt_var = torch.sqrt(var + eps)
    x_hat = (input - mean) / sqrt_var
    out = gamma * x_hat + beta
    ctx.save_for_backward(gamma, x_hat, sqrt_var)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = grad_output.shape[0]

    gamma, x_hat, sqrt_var = ctx.saved_tensors
    grad_input = grad_gamma = grad_beta = None

    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
      # Compute grad_gamma
      grad_gamma = torch.sum(grad_output * x_hat, dim=0)
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
      # Compute grad_beta
      grad_beta = torch.sum(grad_output, dim=0)
    if ctx.needs_input_grad[0]:
      grad_input = batch_size * grad_output - grad_beta - x_hat * grad_gamma
      grad_input = (grad_input * gamma)/(batch_size * sqrt_var)
    ########################
    # END OF YOUR CODE    #
    #######################
    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
