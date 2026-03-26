import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import reshape

# Cauchy-Lipschitz theorem execution

# Hyperparameters
T = 1
d = 2
N = 5
J = 10
Delta_t = T / J

# Dynamics of the ODE

def f(y):
    """Gives the dynamics of the ODE, with vector inputs.
    Inputs:
    - y: Array of shape (d, B), where B is the batch size - Space variable."""
    z = np.zeros_like(y)
    z[0, :] = -y[1, :]
    z[1, :] = y[0, :]
    return z

# Initialization
Y = np.zeros((N+1, d, J+1))
y0 = np.array([1, 0])
Y[0, :, :] =  y0.reshape(d, 1) @ np.ones((1, J+1))

# Mask matrix
cols = np.arange(J+1)          # (J+1,)
levels = np.arange(1, J+2)     # (J,)
mask = cols < -1 + levels[:, None]  # Mask (J, J+1)
M = mask[:, None, :] * np.ones((1, d, 1)) # extend
M = M[:, :, :-1]

# Iterations
for n in range(N):
    F = f(Y[n, :, :])[None, :]
    F = np.repeat(F, J+1, axis=0)