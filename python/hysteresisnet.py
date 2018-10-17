### Script that implements a basic neural network with hysteresis
### W defines the n by n connection matrix between n layers of neurons
### V describes the inital activities of n layers of neurons (each with one neuron for now)

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta))) ### plot this function

class HysteresisNetwork:
    def __init__(self, w, v,dt):
        self.layers    = v
        self.weights   = w
        self.timestep  = dt
        self.lam = 4;
        self.beta = 0.75;
        self.output    = np.zeros(self.layers.shape)
    
    def updatestate(self):
        d_layers = (-1 * self.layers) + sigmoid(np.dot(self.weights, self.layers),self.lam,self.beta)
        self.layers = self.layers + d_layers * self.timestep
        self.output = self.layers



if __name__ == "__main__":
    W = np.array([[0,0,0],
                  [0,0,0],
                  [0,0,0]])
    V = np.array([[0],[0],[0]])
    timestep = 1
    
    hn = HysteresisNetwork(W,V,timestep)
                  
    for i in range(20):
      hn.updatestate()
    print(hn.output)
#plt.ion()
    plt.figure()
    plt.plot(hn.output)



