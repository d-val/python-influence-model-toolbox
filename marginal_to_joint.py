# MARGINAL_TO_JOINT 
# compute the joint (initial) distribution of all states
# input: Pi in bnet, n_states

import numpy as np

def marginal_to_joint(pi, ns):

  pi2 = pi
  pi0 = np.zeros((pi2.shape[0], np.prod(ns)))

  ndx = np.array([0, np.cumsum(ns)])
  ndx = ndx[:-1]
  for i in range(np.prod(ns)):
    s1 = np.unravel_index(i, ns[::-1])
    pi0[:,i] = np.prod(pi2[:,ndx+s1], 1)

  return pi0
