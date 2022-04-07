#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:02:53 2022

@author: bensonb
"""
import numpy as np
import matplotlib.pyplot as plt

duration = 10
n_clusters = 12

times = np.random.rand(200)*duration
clusters = np.random.randint(1,n_clusters+1,size=len(times))

plt.figure(figsize=(3.5,2.5))
plt.plot(times,clusters,'k.')
plt.yticks(np.arange(n_clusters)+1)
plt.ylabel('Cluster indices')
plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('/home/bensonb/IntBrainLab/prior-localization/decoding_figures/cartoons/spikes_to_decoding_matrix.png',
            dpi=200)
plt.show()