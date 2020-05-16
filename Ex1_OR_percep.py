# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:32:18 2020

@author: imrea
"""

### pythonic implementation
# quite a mess, as list cant be multiplied with each other

# implement AND perceptron

X = [[1,1],
     [1,0],
     [0,1],
     [0,0]]

# open output list based on inputs
output = [0] * len(X)

# set weights and threshold.
# for and, you need both input with equal weigth and need to be bigger than one
# value of 2 is the only possibility if both get added

weights = [1,1]
threshold = -1.5

# perceptron
output = []
for pair in X:
    # multiply with weights
    A = pair[0] * weights[0]
    B = pair[1] * weights[1]
    # sum up
    Sum = A+B
    
    # set activation level
    if Sum >= -threshold:
        output.append(1)
    else:
        output.append(0)

print(output)

### numpy implemntation

import numpy as np

# input
X = np.array( ((1,1), (1, 0), (0,1), (0,0)) )

weights = np.array([1,1])
threshold = -1.5

# compute output
net = np.dot(X,weights) # the dotproduct is the easy application
output = np.where(net >= -threshold, 1, 0)
print(output)

























