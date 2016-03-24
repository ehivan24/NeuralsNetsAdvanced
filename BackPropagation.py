
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__="edwingsantos"
__date__ ="$Dec 26, 2014 11:00:53 AM$"


import numpy as np
import matplotlib.pylab as plt

class BackPropagationNetworks:
    """ A Back-Propagation Network """
    
    layerCount = 0
    shape = None
    weights = []
    
    #
    #Class methods
    #
    
    def __init__(self, layerSize):
        """Initialize the Network """
        self.layerCount = len(layerSize)
        self.shape = layerSize
        
        #Input/Output
        self._layerInput = []
        self._layerOutput = []
        
        #create the weights
        for(l1,l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.01, size = (l2, l1+1)))
           
           
    #
    #run Method
    #       
    
    def Run(self, input):
        """ Run the Network"""
        InCases = input.shape[0]
        self._layerInput = []
        self._layerOutput = []
        
        #Run it
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, InCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, InCases])]))   
            
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))
            
            return self._layerOutput[-1].T    
            
    #Transfer Functions
    def sgm(self, x, Derivative=False):  
        if not Derivative:
            return 1/(1+np.exp(-x))
        else:
            out = self.sgm(x)
            return out * (1-out)
              
        
if __name__ == "__main__":
    bpn = BackPropagationNetworks((2,2,2,4,5))
    print(bpn.shape)
    print(bpn.weights)
    
    lvInput = np.array([[0,0],[1,1],[-1, -0.5],[-1,-1],[1,0.5]])
    lvOutput = bpn.Run(lvInput)
    
    print("Input: {0}\n Output: {1}".format(lvInput, lvOutput)) 
    
    
    plt.plot(lvInput)
    plt.plot(lvOutput)
    plt.show()
    
           
    
    
