# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:57:41 2018

@author: Tomas
"""

a = 1000000000
for i in range(1,1000000):
    a = a + 1e-6
a = a - 1000000000
print a