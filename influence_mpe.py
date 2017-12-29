# INFLUENCE_MPE --- find the most probable explanation (state)
# we use gamma since we suspect that a viterbi-like algorithm
# might be NP-hard

import numpy as np
from enter_evidence import *

def influence_mpe(engine, evidence, s = np.array([0])):

  if s.all() == 0:
    s = np.array([np.nan]*evidence.shape[1])
    s = np.array([s]*evidence.shape[0])
  engine, ll = enter_evidence(engine, evidence, s)

  gamma = engine.gamma

  bnet = engine.bnet
  ns = bnet.n_states
  l_ns = len(ns)
  ndx = np.concatenate([[0], np.cumsum(ns)])

  path = np.zeros((evidence.shape[1], l_ns))

  for i in range(l_ns):
    path[:,i] = np.argmax(gamma[:,ndx[i]:ndx[i+1]], axis=1)

  path = path.conj().T
  return path


