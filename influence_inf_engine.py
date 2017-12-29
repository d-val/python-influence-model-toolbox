# INFLUENCE_INF_ENGINE
# An initializer of the influence engine
# call before doing inference

from mk_influence import *
from sample_influence import *
from learn_params_influence import *
from influence_mpe import *
from event_matrix import *
from initialize_mu import *




class engine_c(object):
  def __init__(self, bnet):
    self.nc = bnet.nchains
    self.ns = bnet.n_states
    self.m = bnet.m_outputs
    self.bnet = bnet

def influence_inf_engine(bnet):
  return engine_c(bnet)
