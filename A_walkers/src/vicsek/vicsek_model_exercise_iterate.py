import vicsek_model as vicsek
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
iterations = 1000
N = 2000
v = 0.10
r = 1
L = 32
DL = (0.1, 0.2, 0.4, 0.8, 1.6)
N=int(L**2*DL/r**2)
eta= (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)




for i in range(1,len(DL)-1):
    x, y, orient = vicsek.generate_vicsek_data(iterations=iterations, N=N, eta=eta, v=v, r=r, L=L)
    order=np.sqrt(np.sum(np.cos(orient),axis=1)**2+np.sum(np.sin(orient),axis=1)**2)/N

# plt.scatter(x,y,c=order,map='viridis',s=80,alpha=0.8)

print(N,L,r,r*r*N/L**2,eta,order[iterations-1],iterations)
plt.figure()
plt.plot(order)
plt.show()
