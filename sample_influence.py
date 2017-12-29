# SAMPLE_INFLUENCE generates a random sequence from bnet
# seq are states of hidden nodes at site x and slice y
# obs are symbols observable nodes at site x and slice y

from sample_discrete import *
from sample_gaussian import *
import numpy as np

def sample_influence(bnet, T):
  # to determine the dimension of outputs
  maxout = 1
  for i in range(bnet.nchains):
    if bnet.o_type[i] == 'c' and bnet.m_outputs[i] > maxout:
      maxout = bnet.m_outputs[i]


  seq = np.zeros((bnet.nchains, T)) 
  obs = np.zeros((bnet.nchains, T, maxout)) 
  # sample the first slice
  t = 0
  for i in range(bnet.nchains):
    seq[i, t] = sample_discrete(bnet.Pi[i])
    if bnet.o_type[i] == 'd':
      B = bnet.B[i][seq[i, t]-1, :]
      obs[i, t, 0] = sample_discrete(B)
    else:
      B = bnet.B[i][seq[i,t]-1, :]
      c = bnet.cov[i][seq[i,t]-1,:,:]
      obs[i,t,:] = sample_gaussian(B, c).conj().transpose().flatten()

  # sample the rest slices
  for t in range(1, T):
    for i in range(bnet.nchains):
      # latent state
      T0 = np.zeros(bnet.n_states[i])
      for k in range(bnet.nchains):
        A = bnet.A[k][i]
        T0 = T0 + bnet.T[k,i]*A[seq[k,t-1]-1,:]
      seq[i,t] = sample_discrete(T0)
      # output
      if bnet.o_type[i] == 'd':
        B = bnet.B[i][seq[i,t]-1,:]
        obs[i,t,0] = sample_discrete(B)
      else:
        B = bnet.B[i][seq[i,t]-1,:] #mean
        c = bnet.cov[i][seq[i,t]-1,:,:] #cov
        obs[i,t,:] = sample_gaussian(B,c).conj().transpose().flatten()

  return seq, obs


