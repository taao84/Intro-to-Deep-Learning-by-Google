# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:32:16 2018

@author: Tomas
"""

"""Softmax."""

import numpy as np
from math import exp

#scores = [3.0, 1.0, 0.2]
#scores = [1.0, 2.0, 3.0]
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    shape = np.shape(x)
    numDim = len(shape)
    result = np.zeros_like(x, dtype=float)
    
    if numDim == 1:
        totalSum = 0
        # Sum (e^yi)
        for i in range(len(x)):
            totalSum = totalSum + exp(x[i])
        # e^yi / Sum (e^yi)
        for i in range(len(x)):
            result[i] = exp(x[i])/totalSum
    else:
        xDim = len(x[0])
        totalSum = np.zeros(xDim, dtype=float)
        
        # e^yi
        for i in range(len(x)):
            for j in range(len(x[i])):
                result[i,j] = exp(x[i,j])
                totalSum[j] = totalSum[j] + result[i,j]
        # e^yi / Sum (e^yi)
        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i,j] = result[i,j] / totalSum[j]
    
    return result

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
