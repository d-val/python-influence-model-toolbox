# MARGINAL_TO_JOINT2
# compute the joint (posterior) distribution of states using the influence matrix h
# input: influence matrix h, and n_states

import numpy as np

def marginal_to_joint2(h, ns):
  ndx = np.array([0, np.cumsum(ns[:-1])])
  n = np.prod(ns)
  s1 = np.unravel_index(np.arange(n), ns[::-1])
  s2 = np.unravel_index(np.arange(n), ns[::-1])
  g = np.zeros((n, n))
  for i in range(n):
    h0 = np.sum(h[ndx+s1[i,:],:], 0)
    for j in range(n):
      g[i, j] = np.prod(h0[ndx+s2[j,:]])
  return g
