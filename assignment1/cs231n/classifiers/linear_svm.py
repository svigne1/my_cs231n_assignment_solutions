import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] = dW[:,j] + X[i]
        dW[:,y[i]] = dW[:,y[i]] - X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # For regularization derivative
  dW = dW + 2 * reg * W

  # OMG ERROR.... The way i solved. i confused this 0.002 with 2*h in gradient_check.py file.
  # And then second time confused it with 2 in squared error, that comes out whenever we differentiate.
  dW = (1/num_train) * dW

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  X_scores = X.dot(W)

  # NO LOOP .. FULLY Vectorized form learnt from https://github.com/bruceoutdoors/CS231n
  X_correct_scores = X_scores[np.arange(num_train), y]
  X_scores[np.arange(num_train), y] -=1
  X_correct_scores = np.reshape(X_correct_scores, (-1,1))

  # ONE_LOOP IMPLEMENTATION
  # X_correct_scores = np.zeros((y.shape[0], 1))
  # for i in xrange(y.shape[0]):
  #   X_correct_scores[i][0] = X_scores[i][y[i]]
  #   X_scores[i][y[i]] = X_scores[i][y[i]] - 1

  margin = X_scores - X_correct_scores + 1
  margin /= num_train

  margin_loss = np.sum(margin.clip(min=0))
  regularization_loss = reg * np.sum(W * W)
  loss = margin_loss + regularization_loss


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  boolean_loss = (margin > 0).astype(int)
  n_factor = boolean_loss.dot(np.ones(boolean_loss.shape[1])) * -1

  boolean_loss[np.arange(num_train), y] = boolean_loss[np.arange(num_train), y] + n_factor
  # ^^^ BUG IN THE NOTEBOOK ^^^^
  # print(margin[np.arange(num_train), y])
  # ^^^ notebook is calculates e-19 margin for some correctly classified
  # classes, when ideally it should be 0. Therefore, instead of setting boolean_loss as n_factor. We have used +=
  # in order to avoid error of 58
  bad_classes_gradient = boolean_loss.T.dot(X).T

  dW = bad_classes_gradient

  # SAME ERROR.... Solved it by seeing previous function. 1/numtrain is already done in margin step.
  dW = (1/num_train) * dW

  dW = dW + 2 * reg * W


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
