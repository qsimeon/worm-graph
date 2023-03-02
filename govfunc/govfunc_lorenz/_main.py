#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.1
@file: _main.py
@time: 2023/2/28 10:57
@desc:
'''
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from govfunc.govfunc_lorenz._utils import *

# Generate data
polyorder = 5
usesine = False

sigma = 10
beta = 8/3
rho = 28

n = 3
x0 = np.array([-8, 8, 27])

# Integrate
tspan = np.arange(0, 100, 0.001)
x = odeint(lorenz, x0, tspan, args=(sigma, beta, rho))

# Compute Derivative
eps = 1
dx = np.zeros_like(x)
for i in range(len(x)):
    dx[i] = lorenz(x[i], 0, sigma, beta, rho)
dx = dx + eps * np.random.randn(*dx.shape)

# Pool Data
Theta = poolData(x, n, polyorder, usesine)
m = Theta.shape[1]




# Compute Sparse regression: sequential least squares
lambda_ = 0.025
Xi = sparsifyDynamics(Theta, dx, lambda_, n)

# Plot Lorenz for T in [0, 20]
td = 27
lr = -32
tspan = np.arange(0, 20, 0.001)
xA = odeint(lorenz, x0, tspan, args=(sigma, beta, rho))
xB = odeint(sparseGalerkin, x0, tspan, args=(Xi, polyorder, usesine))

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(xA[:, 0], xA[:, 1], xA[:, 2], 'b', linewidth=1)
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('y', fontsize=13)
ax1.set_zlabel('z', fontsize=13)
ax1.set_title('True Model', fontsize=14)
ax1.view_init(td, lr)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(xB[:, 0], xB[:, 1], xB[:, 2], 'r', linewidth=1)
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel('y', fontsize=13)
ax2.set_zlabel('z', fontsize=13)
ax2.set_title('Sparse Identified Model', fontsize=14)
ax2.view_init(td, lr)

plt.show()