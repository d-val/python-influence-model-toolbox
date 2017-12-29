# ENTER_EVIDENCE --- calculate sufficient statistics from evidence
# since two_slides_evidence might be very large,
# instead of choosing one_slide_evidence and two_slides_evidence
# as sufficient statistics, we use alpha, beta, and obslik .
# 
# returns the influence model engine and log-likelihood value




import numpy as np
import scipy.sparse as sps
import sys
from influence_matrix import *
from gaussian_prob import *
from influence_inf_engine import *


def enter_evidence(engine, evidence, states = np.array([0])):
  bnet = engine.bnet
  ns = bnet.n_states
  m = bnet.m_outputs
  T = np.shape(evidence)[1]
  prior = []
  for i in range(len(bnet.Pi)):
    prior = np.concatenate((prior, bnet.Pi[i]))
  influence = influence_matrix(bnet)


  s_ns = sum(ns)
  l_ns = len(ns)
  ns_1 = np.ones(s_ns)

  ndx = np.concatenate([[0], np.cumsum(ns)])
  if states.all() == 0:
    states = [np.nan]*T
    states = [states]*l_ns
    states = np.asarray(states)
  latent = [np.nan]*T
  latent = [latent]*s_ns
  latent = np.asarray(latent)
  
  t0 = np.arange(T)
  for i in range(l_ns):
    for j in range(T):
      if np.isnan(states[i,j]) != True:
        latent[ndx[i]:ndx[i+1], j] = 0
    ndx0 = ndx[i] + states[i, :]
    for j in range(len(ndx0)):
      if np.isnan(ndx0[j]) == 0:
        latent[ndx0[j],t0[j]] = 1

  obslik = np.zeros((s_ns, T))
  for i in range(l_ns):
    ev = evidence[i, :]
    ndx0 = (np.isnan(ev) == 0)
    if bnet.o_type[i] == 'd':
      for j in range(len(ndx0)):
        if ndx0[j] > 0:
          obslik[ndx[i]:ndx[i+1], j] = bnet.B[i][:,ev[j][0]-1].flatten()
    else:
      for j in range(int(ns[i])):
        M = bnet.B[i][j,:].conj().T
        C = bnet.cov[i][j,:,:]
        for k in range(len(ndx0)):
          if ndx0[k,0] > 0:
            obslik[ndx[i]+j, k] = gaussian_prob(np.mat(ev[k]).T, M, C).conj().T

  obslik[np.isnan(latent) == 0] = latent[np.isnan(latent) == 0]
  obslik = obslik.conj().T + 1e-30   # avoid divide by 0

  ndx9 = np.zeros((s_ns, s_ns))
  for i in range(l_ns):
    ndx9[ndx[i]:ndx[i+1], ndx[i]:ndx[i+1]] = 1

  ndx9 = sps.csr_matrix(ndx9)
  scale = np.ones((T, s_ns))

  alpha = np.zeros((T, s_ns))
  alpha[0,:] = prior * obslik[0,:]
  scale[0,:] = np.asarray(sps.csr_matrix(alpha[0,:]).dot(ndx9).todense())
  alpha[0,:] = alpha[0,:] / scale[0,:]

  for t in range(1, T):
    alpha[t,:] = alpha[t-1, :].dot(influence) * obslik[t,:]
    scale[t,:] = np.asarray(sps.csr_matrix(alpha[t,:]).dot(ndx9).todense())
    alpha[t,:] = alpha[t,:] / scale[t,:]

  ll = np.sum(np.sum(np.log(scale[:,np.asarray(ndx[0:-1], dtype=int)])))

  beta = np.zeros((s_ns, T))
  for i in range(l_ns):
    beta[ndx[i]:ndx[i+1], T-1] = 1/np.sum(alpha[T-1, ndx[i]:ndx[i+1]])
    beta[alpha[T-1, :] == 0, T-1] = 0

  for t in range(T-2, -1, -1):
    beta[:,t] = influence.dot(obslik[t+1,:].conj().T * (beta[:,t+1] / scale[t+1, :].conj().T))

  
  #xi
  mem_tol = 1024*1024
  A = np.zeros(s_ns)
  chunk_sz = pow(s_ns, 2)
  coeff = obslik / scale

  step_sz = 520*int(np.floor(mem_tol/chunk_sz))    # can tune this according to memory availability
  alpha_mat = np.repeat(alpha[:,:, np.newaxis], int(s_ns), axis=2)
  beta_mat = np.repeat(beta[:,:, np.newaxis], int(s_ns), axis=2)
  coeff_mat = np.repeat(coeff[:,:, np.newaxis], int(s_ns), axis=2)

  for t0 in range(0, T-1, step_sz):
    t1 = min(t0+step_sz, T-2)
    t = np.arange(t0, t1+1)

    influence_mat = np.repeat(influence[:,:, np.newaxis], t1-t0+1, axis=2)
    xi = alpha_mat[t,:,:].transpose(1,2,0) * influence_mat 
    xi = xi * (coeff_mat[t+1,:,:].transpose(2,1,0))
    xi = xi * beta_mat[:,t+1,:].transpose(2,0,1)   #two slice marginal
    A = A + np.sum(xi, 2)

  engine.gamma = alpha * beta.conj().T   #one slice marginal
  engine.xi = A   #two slice marginal

  return engine, ll
