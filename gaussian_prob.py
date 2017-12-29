# GAUSSIAN_PROB Evaluate a multivariate Gaussian density.
# p = gaussian_prob(X, m, C)
# p(i) = N(X(:,i), m, C) where C = covariance matrix and each COLUMN of x is a datavector

# p = gaussian_prob(X, m, C, 1) returns log N(X(:,i), m, C) (to prevents underflow).
#
# If X has size dxN, then p has size Nx1, where N = number of examples

import warnings
import numpy as np

def gaussian_prob(x, m, C, use_log = 0):
  if len(m) == 1:  # scalar
    x = x.flatten('F')

  d, N = np.mat(x).shape

  m = m.reshape(np.size(m), 1)
  M = m.dot(np.ones((1, N)))   # replicate the mean across columns
  denom = np.power((2*np.pi), (d/2.0)) * np.sqrt(np.abs(np.linalg.det(C)))
  mahal = np.sum(np.multiply((x - M).conj().T.dot(np.linalg.inv(C)), (x-M).conj().T), 1)   # Chris Bregler's trick
  if np.any(mahal < 0):
    warnings.warn("mahal < 0  => C is not psd", RuntimeWarning)
  if use_log:
    p = -0.5*mahal - np.log(denom)
  else:
    p = np.exp(-0.5*mahal) / (denom+np.spacing(1))

  return p

