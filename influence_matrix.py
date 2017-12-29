# INFLUENCE_MATRIX generates the influence matrix H
# from the influence model representation
# refer Chalee's thesis page 77

import numpy as np

def influence_matrix(bnet):

  nc = bnet.nchains
  ns = bnet.n_states
  A = bnet.A
  T = bnet.T

  ndx = np.concatenate([[0], np.cumsum(ns)])
  sz = np.sum(ns)
  H = np.zeros((sz,sz))

  for i in range(nc):
    for j in range(nc):
      H[ndx[i]:(ndx[i]+ns[i]), ndx[j]:(ndx[j]+ns[j])] = T[i,j]*A[i][j]

  return H
