import vicsek_model as vicsek
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
iterations = 1000
N = 2000
eta = 0.02
v = 0.10
r = 1
L = 32

x, y, orient = vicsek.generate_vicsek_data(iterations=iterations, N=N, eta=eta, v=v, r=r, L=L)
print(x.shape)
print(y.shape)
print(orient.shape)
vicsek.animate_viscek(x, y, orient, iterations)

order=np.sqrt(np.sum(np.cos(orient),axis=1)**2+np.sum(np.sin(orient),axis=1)**2)/N
print(N,L,r,r*r*N/L**2,eta,order[iterations-1],iterations)
plt.figure()
plt.plot(order)
plt.show()
