Dhaval Adjodah <dhaval@mit.edu>, Zheyuan (Ryan) Shi <zshi1@mit.edu>


This python influence model toolbox is translated from the MATLAB toolbox
written by Wen Dong, [which can be found here.](http://vismod.media.mit.edu/vismod/demos/influence-model/)

The influence model is a generative machine learning approach to understand the
influence pattern within a group of entities. Similar to the Hidden
Markov model (HMM), the influence model assumes each entity has an unobservable
latent state as well as an observed state at each time step. HMM uses a combinatorial 
approach to define the distribution characterizing the influence; while the influence model
accomplishes such goal with significantly fewer parameters. Thus, the influence model 
allows for more efficient learning process, and is less likely to overfit. 


Execute the following command or place it at the beginning of the script before
using the influence model toolbox.

```python
from influence_inf_engine import *
```

Let us construct a latent structure influence process with 6 interacting sub-processes, 2 latent states and 2 output symbols per sub-process.

```python
bnet = mk_influence(2*np.ones(6), 2*np.ones(6))
for i in range(bnet.nchains):
  for j in range(bnet.nchains):
    bnet.A[i][j] = np.array([[0.99, 0.01], [0.08, 0.92]])
for i in range(bnet.nchains):
  bnet.B[i] = np.array([[0.9, 0.1], [0.1, 0.9]])  # noise
bnet.T = 0*bnet.T
bnet.T[0:3, 0:3] = 1.0/3 * np.ones((3,3))
bnet.T[3:, 3:] = 1.0/3 * np.ones((3,3))
```

We can sample the latent structure influence process constructed above.
```python
seq, seq0 = sample_influence(bnet,1000)
evidence = seq0 # observed states, seq corresponds to the latent states
```

Parameter learning from the above sample sequence
```python 
bnet1 = mk_influence(2*np.ones(6), 2*np.ones(6))
engine = influence_inf_engine(bnet1)
bnet1, loglik, engine1 = learn_params_influence(engine, seq0, 50)
influence = influence_matrix(bnet1)  # the influence matrix
```

Most probable latent state estimation from the above sample sequence
```python
engine1 = influence_inf_engine(bnet1)
seq2 = influence_mpe(engine1, seq0)

for i in range(seq2.shape[0]):
  if np.sum(seq2[i,:]==0) < np.sum(seq2[i,:]==1):
    seq2[i,:] = 1 - seq2[i,:]

    tmp = influence[2*i,:]
    influence[2*i,:] = influence[2*i+1,:]
    influence[2*i+1,:] = tmp

    tmp = influence[:,2*i]
    influence[:,2*i] = influence[:,2*i+1]
    influence[:,2*i+1] = tmp

fig, ax = plt.subplots(1,1)
ax.imshow = ax.imshow(seq2, interpolation='none', extent=[0,1000,0,6], aspect='auto')
fig.show()
```



Another example with continuous 1-d Gaussian output.

```python
bnet = mk_influence(2*np.ones(6), 1*np.ones(6), 'cccccc')
for i in range(bnet.nchains):
  for j in range(bnet.nchains):
    bnet.A[i][j] = np.array([[.99, .01],[.08, .92]])

for i in range(bnet.nchains):
  bnet.B[i] = np.array([[-.25],[.25]])
  bnet.cov[i] = np.array([np.eye(bnet.m_outputs[i])]*bnet.n_states[i], ndmin=3)

bnet.T = 0*bnet.T
bnet.T[0:3, 0:3] = 1.0/3 * np.ones((3,3))
bnet.T[3:, 3:] = 1.0/3 * np.ones((3,3))

seq, seq0 = sample_influence(bnet,1000) # seq0 is the observed states

bnet1 = mk_influence(2*np.ones(6), 1*np.ones(6),'cccccc')
bnet1 = initialize_mu(bnet1,seq0)
engine = influence_inf_engine(bnet1)
bnet1,ll,engine1 = learn_params_influence(engine,seq0,50)
seq2 = influence_mpe(engine1,seq0)
```

Development environment:
- python 2.7.12
- numpy 1.11.0
- matplotlib 1.5.1
- scipy 0.17.0
