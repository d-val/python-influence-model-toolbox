# JOINT_TO_MARGINAL
# compute the marginal (initial) distribution of each states
# input: joint distribution of Pi in bnet, n_states
# inverse of MARGINAL_TO_JOINT
import numpy as np
def joint_to_marginal(pi0, ns):
  pi2 = list()
  for i in range(len(ns)):
    pi2.append(np.zeros((pi0.shape[0], ns[i])))

  ndx = np.array([0, np.cumsum(ns)])
  ndx = ndx[0:-1]
  for i in range(np.prod(ns)):
    for j in range(len(ns)):
      s1 = np.unravel_index(i, ns[::-1])
      pi2[j][:,s1[j]] = pi2[j][:,s1[j]] + pi0[:,i]
  return pi2

