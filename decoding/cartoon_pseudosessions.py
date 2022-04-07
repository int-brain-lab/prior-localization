#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:02:53 2022

@author: bensonb
"""

import numpy as np
import matplotlib.pyplot as plt
from brainbox.task.closed_loop import generate_pseudo_blocks

n_samples = 10

plt.figure(figsize=(3.5,2.5))
for ni in range(n_samples):
    session_duration = 700#np.random.randint(400,1000)
    y = generate_pseudo_blocks(session_duration, factor=60, min_=20, max_=100, first5050=90)
    plt.plot(np.arange(session_duration), (y*0.6)+ni,'k-')
plt.yticks(np.arange(n_samples)+1, [])
plt.ylabel('Samples')
plt.xlabel('Trials')
plt.tight_layout()
plt.savefig('/home/bensonb/IntBrainLab/prior-localization/decoding_figures/cartoons/pseudosessions.png',
            dpi=200)
plt.show()