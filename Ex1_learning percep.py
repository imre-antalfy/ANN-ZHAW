# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:32:18 2020

@author: imrea
"""

from Perceptron_fun import Perceptron
import numpy as np

X = [[1,1],
     [1,0],
     [0,1],
     [0,0]]

weights = [0,0]

# threshold = -1.5 threshold is deactivated, the perceptron should learn it
# instead use ground truth
GT = [1,0,0,0]


net = np.dot(X,weights) # the dotproduct is the easy application
if net == GT: # if output met GT, weights are learned well
    output = net 
else: # change weights
    

# output = np.where(net >= -threshold, 1, 0)


print(output)










res = Perceptron(X,weights,threshold)












