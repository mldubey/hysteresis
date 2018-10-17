### Script that implements a basic neural network with hysteresis
### W defines the n by n connection matrix between n layers of neurons
### V describes the inital activities of n layers of neurons (each with one neuron for now)

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta))) ### plot this function

class HysteresisNetwork:
    def __init__(self, w, v, lam, beta, dt):
        self.layers    = v
        self.weights   = w
        self.timestep  = dt
        self.lam = lam;
        self.beta = beta;
        self.output    = np.zeros(self.layers.shape)
    
    def updatestate(self):
        d_layers = (-1 * self.layers) + sigmoid(np.dot(self.weights, self.layers),self.lam,self.beta)
        self.layers = self.layers + d_layers * self.timestep
        self.output = self.layers

## 1's are perfect integrators
## <1 is a leaky integrator
## >1 is a switch

W = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])
V = np.array([[0.5],[0.25],[0]])

lam = 4;
beta = .5;
timestep = 1

hn = HysteresisNetwork(W,V,lam,beta,timestep)

for i in range(20):
  hn.updatestate()
print(hn.output)

plt.ion()
plt.figure()
plt.plot(hn.output)



