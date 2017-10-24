import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    raw_scores = X[i].dot(W)

    # Solving numerical instability... shifting raw_scores
    raw_scores -= np.max(raw_scores)
    softmax_scores = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
    loss += -np.log(softmax_scores[y[i]])

    softmax_scores[y[i]] += -1
    dY = np.tile(softmax_scores, (W.shape[0], 1))
    dW = dW + dY * np.tile(np.reshape(X[i], (-1,1)), (1, W.shape[1]))


  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW /= num_train
  # For regularization derivative
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  raw_scores = X.dot(W)
  raw_scores = raw_scores - np.reshape(np.max(raw_scores,axis=1), (-1,1))
  softmax_scores = np.exp(raw_scores)
  softmax_scores /= np.reshape(np.sum(softmax_scores, axis=1), (-1,1))

  loss = -np.sum(np.log(softmax_scores[np.arange(softmax_scores.shape[0]),y]))
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  softmax_scores[np.arange(softmax_scores.shape[0]),y] -= 1
  dW = softmax_scores.T.dot(X).T

  dW /= num_train
  # For regularization derivative
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
