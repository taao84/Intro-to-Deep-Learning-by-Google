# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:32:16 2018

@author: Tomas
"""

"""Softmax."""

import numpy as np

#scores = [3.0, 1.0, 0.2]
#scores = [1.0, 2.0, 3.0]
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    result = np.exp(x) / np.sum(np.exp(x), axis=0)
    
    return result

softMaxScores = softmax(scores)
print(softMaxScores)

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores2 = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
scores2 = scores2;
softMaxScores = softmax(scores2)

plt.plot(x, softMaxScores.T, linewidth=2)
plt.show()

softMaxScoresSum = np.sum(softMaxScores, axis=0)
objects = list(range(0, len(softMaxScoresSum), 1))
plt.bar(objects, softMaxScoresSum, align='center', alpha=0.5)
plt.show()

