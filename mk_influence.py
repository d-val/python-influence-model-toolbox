# MK_INFLUENCE --- construct an influence model dbn
#
# input:
# n_states --- number of latent states for each chain (1 x n_chains) 
# m_outputs --- for discrete output, number of output symbols
#               for continous output, dimensionality  (1 x n_chains)
# o_type --- output type, 'c' continuous, 'd' discrete (1 x n_chains)
# the chains can have different number of latent states and output
#
# n_states contains the number of internal states for each chain

import sys
import numpy as np

class bnet_c(object):
    def __init__(self, nchains, n_states, m_outputs, o_type):
        self.nchains = nchains
        self.n_states = n_states
        self.m_outputs = m_outputs
        self.o_type = o_type
        



def mk_influence(n_states, m_outputs, o_type = -1):
  if o_type == -1:
    o_type = ['d']*len(n_states)
  n_states = n_states.astype(int)
  m_outputs = m_outputs.astype(int)
  bnet = bnet_c(len(n_states), n_states, m_outputs, o_type)


  #<initial distribution for each hidden node in each. C sites
  bnet.Pi = list()
  for i in range(bnet.nchains):
      Pi = np.random.rand(int(n_states[i]))
      bnet.Pi.append(Pi/sum(Pi))

  #output matrix for each output node in each. C sites
  bnet.B = [None]*bnet.nchains
  bnet.cov = [None]*bnet.nchains
  for i in range(bnet.nchains):
      if o_type[i] == 'd':
          B = np.random.rand(n_states[i], m_outputs[i]).T
          bnet.B[i] = B/np.repeat(np.sum(B, 1, keepdims=True), m_outputs[i], 1)
      elif o_type[i] == 'c':
          bnet.B[i] = np.random.randn(n_states[i], m_outputs[i]).T
          bnet.cov[i] = 100*np.array([np.eye(m_outputs[i])]*n_states[i], ndmin=3)
      else:
          raise ValueError('invalid output')

  #C^2 transition matrix for every 2 hidden nodes in different sites
  #A is 4-dimensional: (site m node i, site n node j)
  bnet.A = list()
  for i in range(bnet.nchains):
      bnet.A.append(list())
      for j in range(bnet.nchains):
          A = np.random.rand(n_states[i], n_states[j]).T
          bnet.A[i].append(A/np.repeat(np.sum(A, 1, keepdims=True), n_states[j], 1))

  # C coupling parameters for every 2 sites
  T = np.random.rand(bnet.nchains, bnet.nchains).T
  bnet.T = T/np.repeat(np.sum(T, 0, keepdims=True), bnet.nchains, 0)

  return bnet




