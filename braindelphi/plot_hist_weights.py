#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:45:03 2023

@author: bensonb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATES = ['01-04-2023', 
         '01-04-2023', 
         '01-04-2023', 
         '01-04-2023', 
         '01-04-2023', 
         '01-04-2023',]
VARIS = ['block', 
         'stimside', 
         'choice', 
         'feedback', 
         'wheel-speed', 
         'wheel-vel']

all_ws = []
for DATE, VARI in zip(DATES, VARIS):
    ws = pd.read_csv(
        f'decoding_processing/{DATE}_{VARI}_clusteruuids_weights.csv')
    ws = ws.filter(regex = 'ws*').values.flatten()
    all_ws.append(np.copy(np.abs(ws)))

max_w = np.max([np.max(ws) for ws in all_ws])
for ws in all_ws:
    plt.hist(ws, bins=np.linspace(0,max_w,1001), histtype='step', density=True)
    
plt.legend([f'{v} ({np.mean(all_ws[i]==0):.3f})' for i, v in enumerate(VARIS)], 
           fontsize=11,
           title='Variable (frac. equal to 0)')
plt.yscale('log')    
plt.xscale('log')
plt.ylabel('Density')
plt.xlabel('Weights')
plt.savefig('decoding_figures/weight_hist.svg')
plt.show()

