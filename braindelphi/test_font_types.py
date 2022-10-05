#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:11:14 2022

@author: bensonb
"""
#test font types

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.rcParams['font.sans-serif'])
print(matplotlib.rcParams['font.family'])
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
matplotlib.rcParams['font.family'] = 'font.cursive'

plt.title('This is Awesome')
plt.plot(np.random.rand(10))
plt.xlabel('Hello')
plt.ylabel('Good game')
plt.show()