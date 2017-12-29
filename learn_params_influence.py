# LEARN_PARAMS_INFLUENCE - learn the influence model parameters using EM
# user specifies number of iterations
# returned engine contains the learned parameters

import matplotlib.pyplot as plt
from enter_evidence import *
from maximize_params import *
from influence_matrix import *

def learn_params_influence(engine, ev, iter, s = np.array([0])):
  plt.interactive(True)
  loglik = []
  verbose = 1

  if s.all() == 0:
    s = np.array([np.nan]*ev.shape[1])
    s = np.array([s]*ev.shape[0])

  n = np.sum(engine.ns)
  fig, ax = plt.subplots(1, 1)
  for i in range(iter):
    engine, ll = enter_evidence(engine, ev, s)
    if verbose:
      print "EM iteration %(iter)d, ll = %(val)8.8f\n" %{"iter": i+1, "val": ll}
    loglik = np.concatenate([loglik, [ll]])
    engine.bnet = maximize_params(engine, ev)
    # termination criterion
    if i > 50:
      if ll-loglik[i-1] < 0.1 and loglik[i-1]-loglik[i-2] < 0.1:
        break

    ax.imshow(influence_matrix(engine.bnet), interpolation='none', extent=[0, n, 0, n])
    
    fig.canvas.draw()

  bnet = engine.bnet
  return bnet, loglik, engine
    
    

