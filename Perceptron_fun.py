# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:32:18 2020

@author: imrea
"""

### perceptron function

import numpy as np

# compute output
def Perceptron(X,weights,threshold):
    net = np.dot(X,weights) # the dotproduct is the easy application
    output = np.where(net >= -threshold, 1, 0)
    return(output)















