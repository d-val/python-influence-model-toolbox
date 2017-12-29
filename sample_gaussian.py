
# SAMPLE_GAUSSIAN Draw N random row vectors from a Gaussian distribution

# samples = sample_gaussian(mean, cov, N)

# copied from Kevin Murphy's BNT package


import numpy as np
def sample_gaussian(mu, Sigma, N = 1):

# If Y = CX, Var(Y) = C Var(X) C'.

# So if Var(X)=I, and we want Var(Y)=Sigma, we need to find C. s.t. Sigma = C C'.

# Since Sigma is psd, we have Sigma = U D U' = (U D^0.5) (D'^0.5 U').

  mu = np.asarray(mu)
  Sigma = np.asarray(Sigma)
  mu = mu.reshape(np.size(mu), 1)
  n = len(mu)
  U, d, V = np.linalg.svd(Sigma)
  D = np.diag(d)
  V = V.T
  M = np.random.randn(n, N)
  M = (np.mat(U)*np.sqrt(np.mat(D)))*np.mat(M) + mu*np.ones((1,N)) # transform each column
  M = M.conj().transpose()

  return M

