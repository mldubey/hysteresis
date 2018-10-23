### Script that implements a basic neural network with hysteresis
### W defines the n by n connection matrix between n layers of neurons
### V describes the inital activities of n layers of neurons (each with one neuron for now)

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta))) ### plot this function

class HysteresisNetwork:
    def __init__(self, w, v, lam, beta, dt, ext_input):
        self.layers    = v
        self.weights   = w
        self.ext_input = ext_input
        self.timestep  = dt
        self.lam = lam;
        self.beta = beta;
        self.output    = np.zeros(self.layers.shape)
    
    def updatestate(self):
        int_input = np.dot(self.weights, self.layers)  ### TODO: Replace with Thomas's Matrix Mutliply
        input = self.ext_input + int_input
        d_layers = (-1 * self.layers) + sigmoid(input,self.lam,self.beta)
        self.layers = self.layers + d_layers * self.timestep
        self.output = self.layers


## 1's are perfect integrators
## <1 is a leaky integrator
## >1 is a switch
## Specify the desired params here

W = np.array([0.5])
V = np.array([1])

lam = 4;
beta = 0.5;
timestep = 0.01

itr = 0

ext_input = [0] * 201
output = [0] * 201

for m in range(-100,100,1):
    
    itr = itr + 1
    ext_input[itr] = float(m) / 100;
    hn = HysteresisNetwork(W,V,lam,beta,timestep,ext_input[itr])
    
    for i in range(1000):
      hn.updatestate()
    
    output[itr] = hn.output

    print(hn.output)


# plot output
plt.ion
plt.plot(ext_input, output)
plt.axis([-1,1,0,1])
plt.xlabel('External Input Level')
plt.ylabel('Output Level (V)')
plt.title('Network Response Curve')

plt.show()
