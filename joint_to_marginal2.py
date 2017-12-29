# JOINT_TO_MARGINAL2
# compute the marginal (posterior) distribution of states using joint distribution
# input: joint distribution g from influence matrix h, and n_states
# inverse of MARGINAL_TO_JOINT2
import numpy as np
from mk_influence import *
from event_matrix import *
def joint_to_marginal2(g, ns):
  bnet = bnet_c()
  bnet.nchains = len(ns)
  bnet.n_states = ns
  B = event_matrix(bnet)
  h = np.linalg.pinv(B).dot(g.dot(B))
  return h
