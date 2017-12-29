# MASTER_MARKOV_MATRIX - calculates the G in Chalee's thesis formula 5.11

from influence_matrix import *
from marginal_to_joint2 import *
from mk_influence import *
def master_markov_matrix(bnet):
  h = influence_matrix(bnet)
  g = marginal_to_joint2(h, bnet.n_states)
  return g
