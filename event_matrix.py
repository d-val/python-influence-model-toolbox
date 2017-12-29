# EVENT_MATRIX generates the event matrix B
# from the influence model representation
# refer Chalee's thesis

import numpy as np

def event_matrix(bnet):

  nc = bnet.nchains
  ns = bnet.n_states

  ndx = np.concatenate([[0], np.cumsum(ns)])
  st3p = np.cumprod(ns[nc-1::-1])
  st3p = np.concatenate([st3p[nc-2::-1], [1]])
  sz1 = np.prod(ns)
  sz = sz1/ns
  sz2 = np.sum(ns)
  B = np.zeros((sz1, sz2))

  for c in range(nc):
    fil1 = np.eye(ns[c])
    fil1 = np.asarray([fil1]*sz[c])
    fil1 = fil1.transpose(1,2,0)
    fil1 = np.reshape(fil1, (ns[c], ns[c], st3p[c], sz[c]/st3p[c]))
    fil1 = fil1.transpose(2,0,3,1)
    fil1 = np.reshape(fil1, (sz1, ns[c]), order='F')
    B[:, ndx[c]:(ndx[c]+ns[c])] = fil1


  D1 = np.ones(sz2)
  D2 = np.zeros(sz2)
  denon = 0
  for c in range(nc):
    i = np.concatenate([np.arange(0, c), np.arange(c+1, nc)])
    D1[ndx[c]:(ndx[c]+ns[c])] = 1 + ns[c] * sum(1/ns[i])
    D2[ndx[c]:(ndx[c]+ns[c])] = -sum(1/ns[i])
    denon = denon + np.prod(ns[i])

  D1 = D1/denon
  D2 = D2/denon
  pinvB = np.diag(D1).dot(B.conj().T) + np.diag(D2).dot(np.ones((sz2, sz1)))

  return B, D1, D2, pinvB
