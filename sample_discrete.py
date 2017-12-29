# SAMPLE_DISCRETE Like the built in 'rand', except we draw from a non-uniform discrete distrib.
# M = sample_discrete(prob, r, c)

# Example: sample_discrete([0.8 0.2], 1, 10) generates a row vector of 10 random integers from {1,2},
# where the prob. of being 1 is 0.8 and the prob of being 2 is 0.2.

# copied from Kevin Murphy's BNT package

import numpy as np
import sys
def sample_discrete(prob, r = 1, c = -1):

  
  n = len(prob)
  
  if c == -1:
    c = r
  
  R = np.random.rand(r, c)
  M = np.ones((r, c))
  cumprob = np.cumsum(prob)

  if n < r*c:
    for i in range(n-1):
      M = M + (R > cumprob[i])
  else:
    #loop over the smaller index - can be much faster if len(prob) >> r*c
    cumprob2 = cumprob[:-1]
    for i in range(r):
      for j in range(c):
        M[i,j] = np.sum(R[i, j] > cumprob2) + 1

  return M



