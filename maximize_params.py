# MAXIMIZE_PARAMS maximize dbn parameters according to
# sufficient statistics we collected

import numpy as np
import scipy.sparse as sps
import scipy as sp
from influence_inf_engine import *
from mk_influence import *
from influence_matrix import *

def maximize_params(engine, evidence):

  bnet = engine.bnet
  ns = bnet.n_states
  m = bnet.m_outputs
  T = evidence.shape[1]
  influence = influence_matrix(bnet)


  s_ns = sum(ns)
  l_ns = len(ns)
  ns_1 = np.ones(s_ns)


  ndx = np.concatenate([[0], np.cumsum(ns)])
  ndx9 = np.zeros((s_ns, s_ns))
  for i in range(l_ns):
    ndx9[ndx[i]:ndx[i+1], ndx[i]:ndx[i+1]] = 1

  ndx9 = sps.csr_matrix(ndx9)

  gamma = engine.gamma

  pi = gamma[0,:] / np.asarray(sps.csr_matrix(gamma[0,:]).dot(ndx9).todense()).flatten()
  for c1 in range(l_ns):
    engine.bnet.Pi[c1] = pi[ndx[c1]:ndx[c1+1]]

  mem_tol = 1024*1024
  A = engine.xi

  for c1 in range(l_ns):
    for c2 in range(l_ns):
      tmp = A[ndx[c1]:ndx[c1+1], ndx[c2]:ndx[c2+1]] + np.spacing(1)
      engine.bnet.A[c1][c2] = tmp / np.repeat(np.sum(tmp, 1, keepdims=True), tmp.shape[1], 1)

  coeff2 = np.zeros((l_ns, s_ns))
  for i in range(l_ns):
    coeff2[i, ndx[i]:ndx[i+1]] = 1

  tmp = coeff2.dot(A.dot(coeff2.conj().T))
  engine.bnet.T = tmp / np.repeat(np.sum(tmp, 0, keepdims=True), tmp.shape[0], 0)


  cov_prior = 0.01
  coeff2 = np.arange(1, max(ns)+1)
  coeff2 = np.asarray([coeff2]*T)
  for i in range(l_ns):
    
    
    if bnet.o_type[i] == 'd':
      ev = evidence[i,:].flatten()
      ndx0 = (np.isnan(ev) == 0)
      engine.bnet.B[i] = sps.csr_matrix((gamma[ndx0, ndx[i]:ndx[i+1]].flatten(), (coeff2[ndx0, 0:ns[i]].flatten()-1, np.asarray([ev[ndx0 > 0]]*ns[i]).conj().T.flatten()-1)), shape = (ns[i], m[i])).todense()
      tmp = engine.bnet.B[i]
      tmp = np.asarray(tmp)
      engine.bnet.B[i] = tmp / np.repeat(np.sum(tmp, 1, keepdims=True), tmp.shape[1], 1)

    else:
      ev = evidence[i,:].T
      ndx0 = (np.isnan(np.sum(ev, 0)) == 0).flatten()
      for j in range(int(ns[i])):
        coeff = gamma[:,ndx[i]+j].conj().T
        coeff = coeff/np.sum(coeff[ndx0])
        m_1 = np.ones(m[i])
        engine.bnet.B[i][j,:] = np.sum(np.asarray([coeff[ndx0]]*m[i]) * ev[:,ndx0], 1).conj().T

        engine.bnet.cov[i][j,:,:] = np.zeros((m[i], m[i]))
        chunk_sz = pow(m[i], 2)
        step_sz = 520*int(np.floor(mem_tol/chunk_sz))  # can tune this according to memory availability
        for t0 in range(0, T, step_sz):
          t1 = np.minimum(t0 + step_sz-1, T)
          t = np.arange(t0, t1)
          t = t[ndx0[t]]
          engine.bnet.cov[i][j,:,:] = engine.bnet.cov[i][j,:,:] + np.sum(np.asarray([ev[:,ndx0>0]]*m[i]) * np.asarray([ev[:,ndx0]]*m[i]).transpose(1,0,2) * np.reshape(np.asarray([[(coeff.T)[t]]*m[i]]*m[i]), (m[i], m[i], len(t))), 2)

        engine.bnet.cov[i][j,:,:] = engine.bnet.cov[i][j,:,:] - (engine.bnet.B[i][j,:].conj().T).dot(engine.bnet.B[i][j,:]) 
        #hack, add some prior
        engine.bnet.cov[i][j,:,:] = engine.bnet.cov[i][j,:,:] + cov_prior*np.eye(m[i])

  bnet = engine.bnet
  return bnet

