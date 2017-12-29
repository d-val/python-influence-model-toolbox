# INITIALIZE_MU --- initialize mean for continous chains
# we hope to accelerate convergence and 
# put centroids to better locations in this way

import scipy.cluster.vq as cl
from mk_influence import *
def initialize_mu(bnet, ev):

  ns = bnet.n_states
  m = bnet.m_outputs
  l_ns = len(ns)

  for i in range(l_ns):
    bnet.B[i], dist = cl.kmeans(np.mat(ev[i,:]), ns[i])
    while bnet.B[i].shape[0] < ns[i]:
      bnet.B[i] = np.mat(np.append(bnet.B[i], np.mat(ev[i,:][0]), axis=0))

  return bnet

